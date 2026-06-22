import os
import tempfile

import numpy as np
from PIL import Image

from biased_split.molecularnetwork import (
    smiles_to_ecfp4_bitvect,
    compute_similarity_matrix,
    molecular_network_from_list,
    visualise_molnet_split,
)

UNASSIGNED_NODE = 0
TRAIN_NODE = 1
TEST_NODE = 2


class KNNFailureSplitter:
    def __init__(
        self, similarity_threshold, activity_threshold, n_neighbors, test_fraction=0.2
    ):
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.n_neighbors = n_neighbors
        self.test_fraction = test_fraction

    def split_for_intended_bias(
        self, smiless, similarity_matrix, activity_values, intended_bias, random_seed
    ):
        if not (0.0 <= intended_bias <= 1.0):
            raise ValueError(f"intended_bias must be in [0, 1], got {intended_bias}")

        rng = np.random.default_rng(random_seed)
        n_molecules = len(smiless)
        target_test_size = int(self.test_fraction * n_molecules)
        n_failure_test_target = int(intended_bias * target_test_size)

        failure_n_edges = self.find_failure_n_edges(
            similarity_matrix,
            activity_values,
            self.similarity_threshold,
            self.activity_threshold,
            self.n_neighbors,
        )
        shuffled_order = rng.permutation(len(failure_n_edges))
        failure_n_edges = [failure_n_edges[i] for i in shuffled_order]

        assignment = self.walk_failure_n_edges(
            failure_n_edges, n_molecules, n_failure_test_target
        )

        candidate_set = {molecule_index for molecule_index, _ in failure_n_edges}
        is_candidate_mask = np.zeros(n_molecules, dtype=bool)
        if candidate_set:
            is_candidate_mask[list(candidate_set)] = True

        unassigned_indices = np.where(assignment == UNASSIGNED_NODE)[0]
        unassigned_non_candidate_indices = unassigned_indices[
            ~is_candidate_mask[unassigned_indices]
        ]
        unassigned_candidate_indices = unassigned_indices[
            is_candidate_mask[unassigned_indices]
        ]

        n_random_fill = target_test_size - int((assignment == TEST_NODE).sum())
        if n_random_fill > 0:
            if len(unassigned_non_candidate_indices) >= n_random_fill:
                random_test_indices = rng.choice(
                    unassigned_non_candidate_indices, size=n_random_fill, replace=False
                )
            else:
                shortfall = n_random_fill - len(unassigned_non_candidate_indices)
                candidate_topup_indices = rng.choice(
                    unassigned_candidate_indices,
                    size=min(shortfall, len(unassigned_candidate_indices)),
                    replace=False,
                )
                random_test_indices = np.concatenate(
                    [unassigned_non_candidate_indices, candidate_topup_indices]
                )
            assignment[random_test_indices] = TEST_NODE

        assignment[assignment == UNASSIGNED_NODE] = TRAIN_NODE

        train_indices = np.where(assignment == TRAIN_NODE)[0]
        test_indices = np.where(assignment == TEST_NODE)[0]

        question_results = self.evaluate_knn_failure_question(
            test_indices,
            train_indices,
            np.asarray(activity_values, dtype=float),
            similarity_matrix,
            self.similarity_threshold,
            self.activity_threshold,
            self.n_neighbors,
        )
        effective_bias = self.effective_bias_from_question_results(question_results)

        return train_indices, test_indices, effective_bias

    def split(self, smiless, activity_values, intended_biases, n_repeats):

        fps_bitvect = [smiles_to_ecfp4_bitvect(s) for s in smiless]
        similarity_matrix = compute_similarity_matrix(fps_bitvect)

        for intended_bias in intended_biases:
            for repeat_index in range(n_repeats):
                train_indices, test_indices, effective_bias = (
                    self.split_for_intended_bias(
                        smiless=smiless,
                        similarity_matrix=similarity_matrix,
                        activity_values=activity_values,
                        intended_bias=intended_bias,
                        random_seed=repeat_index,
                    )
                )
                yield train_indices, test_indices, effective_bias, intended_bias, repeat_index

    @staticmethod
    def find_failure_n_edges(
        similarity_matrix,
        activity_values,
        similarity_threshold,
        activity_threshold,
        n_neighbors,
    ):
        n_molecules = len(activity_values)
        n_edges = []
        for molecule_index in range(n_molecules):
            similarities = similarity_matrix[molecule_index].copy()
            similarities[molecule_index] = -1.0
            qualifying = np.where(similarities >= similarity_threshold)[0]
            if len(qualifying) < n_neighbors:
                continue
            top_k = qualifying[np.argsort(similarities[qualifying])[::-1][:n_neighbors]]
            consensus = float(activity_values[top_k].mean())
            disagreement = abs(consensus - float(activity_values[molecule_index]))
            if disagreement >= activity_threshold:
                n_edges.append((int(molecule_index), tuple(int(n) for n in top_k)))
        return n_edges

    @staticmethod
    def walk_failure_n_edges(failure_n_edges, n_molecules, n_failure_test_target):
        assignment = np.full(n_molecules, UNASSIGNED_NODE, dtype=np.int8)
        n_failures_placed = 0
        for molecule_index, neighbor_indices in failure_n_edges:
            if n_failures_placed >= n_failure_test_target:
                break
            if assignment[molecule_index] == TRAIN_NODE:
                continue
            if any(assignment[n] == TEST_NODE for n in neighbor_indices):
                continue
            assignment[molecule_index] = TEST_NODE
            for neighbor_index in neighbor_indices:
                assignment[neighbor_index] = TRAIN_NODE
            n_failures_placed += 1
        return assignment

    @staticmethod
    def evaluate_knn_failure_question(
        test_indices,
        train_indices,
        activity_values,
        similarity_matrix,
        similarity_threshold,
        activity_threshold,
        n_neighbors,
    ):
        results = np.full(len(test_indices), np.nan, dtype=float)
        if len(test_indices) == 0 or len(train_indices) == 0:
            return results
        for position, test_idx in enumerate(test_indices):
            similarities_to_train = similarity_matrix[test_idx][train_indices]
            qualifying = np.where(similarities_to_train >= similarity_threshold)[0]
            if len(qualifying) < n_neighbors:
                continue
            top_k_positions = qualifying[
                np.argsort(similarities_to_train[qualifying])[::-1][:n_neighbors]
            ]
            top_k_train_indices = train_indices[top_k_positions]
            consensus = float(activity_values[top_k_train_indices].mean())
            disagreement = abs(consensus - float(activity_values[test_idx]))
            results[position] = 1.0 if disagreement >= activity_threshold else 0.0
        return results

    @staticmethod
    def effective_bias_from_question_results(question_results):
        if question_results.size == 0:
            return 0.0
        evaluable = question_results[~np.isnan(question_results)]
        if evaluable.size == 0:
            return 0.0
        return float(evaluable.mean())

    def visualise_splits(
        self,
        smiless,
        activity_values,
        intended_biases,
        n_repeats,
        output_path,
        duration=500,
    ):
        G = molecular_network_from_list(
            smiless, activity_values, self.similarity_threshold, self.activity_threshold
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for frame_index, (
                train_idx,
                test_idx,
                effective_bias,
                intended_bias,
                _,
            ) in enumerate(
                self.split(smiless, activity_values, intended_biases, n_repeats)
            ):
                p = os.path.join(tmpdir, f"frame_{frame_index:04d}.png")
                visualise_molnet_split(
                    G, train_idx, test_idx, effective_bias, intended_bias, filepath=p
                )
                paths.append(p)
            frames = [Image.open(p) for p in paths]
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )

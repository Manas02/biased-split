import os
import tempfile
from PIL import Image
import numpy as np

from biased_split.molecularnetwork import (
    smiles_to_ecfp4_bitvect,
    compute_similarity_matrix,
    molecular_network_from_list,
    visualise_molnet_split,
)

UNASSIGNED_NODE = 0
TRAIN_NODE = 1
TEST_NODE = 2


class ActivityCliffSplitter:
    def __init__(
        self,
        similarity_threshold,
        activity_threshold,
        test_fraction=0.2,  # of total dataset, default 20% of total dataset should be test set
    ):
        self.similarity_threshold = similarity_threshold
        self.activity_threshold = activity_threshold
        self.test_fraction = test_fraction

    def split_for_intended_bias(
        self,
        smiless,
        similarity_matrix,
        activity_values,
        intended_bias,  # this is the fraction that we _try_ to construct. Depending on dataset and parameters this may not be possible and thus we ALWAYS report and use *effective bias*.
        random_seed,
    ):
        if not (0.0 <= intended_bias <= 1.0):
            raise ValueError(f"intended_bias must be in [0, 1], got {intended_bias}")

        rng = np.random.default_rng(random_seed)
        n_molecules = len(smiless)
        # int(2.1) => 2; int(2.9) => 2; thus int here acts as floor operator
        target_test_size = int(self.test_fraction * n_molecules)
        n_cliff_test_molecules = int(intended_bias * target_test_size)

        cliff_edges = self.find_cliff_edges(
            similarity_matrix=similarity_matrix,
            activity_values=activity_values,
            similarity_threshold=self.similarity_threshold,
            activity_threshold=self.activity_threshold,
        )  # this gives us (node idx1, node idx2, activity difference)

        # One can sort edges so the largest activity gaps are processed first. But in this case, we will randomly sort it.
        # cliff_edges.sort(key=lambda edge: edge[2], reverse=True) # edge[2] is the activity difference from cliff_edges
        rng.shuffle(cliff_edges)

        # calculate cliff degrees for heuristic sorting into TRAIN_NODE
        cliff_degrees = self.compute_cliff_degrees(cliff_edges, n_molecules)

        # assign the cliff nodes by walking the cliff edges
        assignment = self.walk_cliff_edges(
            cliff_edges=cliff_edges,
            cliff_degrees=cliff_degrees,
            n_molecules=n_molecules,
            n_cliff_test_target=n_cliff_test_molecules,
            rng=rng,
        )

        unassigned_indices = np.where(assignment == UNASSIGNED_NODE)[0]
        unassigned_non_cliff_indices = unassigned_indices[
            cliff_degrees[unassigned_indices] == 0
        ]
        unassigned_cliff_indices = unassigned_indices[
            cliff_degrees[unassigned_indices] > 0
        ]

        n_random_fill = target_test_size - int((assignment == TEST_NODE).sum())

        if n_random_fill > 0:
            if len(unassigned_non_cliff_indices) >= n_random_fill:
                random_test_indices = rng.choice(
                    unassigned_non_cliff_indices, size=n_random_fill, replace=False
                )
            else:
                shortfall = n_random_fill - len(unassigned_non_cliff_indices)
                cliff_topup_indices = rng.choice(
                    unassigned_cliff_indices,
                    size=min(shortfall, len(unassigned_cliff_indices)),
                    replace=False,
                )
                random_test_indices = np.concatenate(
                    [unassigned_non_cliff_indices, cliff_topup_indices]
                )
            assignment[random_test_indices] = TEST_NODE

        # now, all unassigned molecules go to training.
        assignment[assignment == UNASSIGNED_NODE] = TRAIN_NODE

        train_indices = np.where(assignment == TRAIN_NODE)[0]
        test_indices = np.where(assignment == TEST_NODE)[0]

        question_results = self.evaluate_cliff_question(
            test_indices=test_indices,
            train_indices=train_indices,
            similarity_matrix=similarity_matrix,
            activity_values=activity_values,
            similarity_threshold=self.similarity_threshold,
            activity_threshold=self.activity_threshold,
        )

        # calculate the effective bias after random sampling.
        effective_bias = self.effective_bias_from_question_results(question_results)
        return train_indices, test_indices, effective_bias

    def split(self, smiless, activity_values, intended_biases, n_repeats):
        fps_bitvect = [smiles_to_ecfp4_bitvect(smiles) for smiles in smiless]
        similarity_matrix = compute_similarity_matrix(fps_bitvect)

        for intended_bias in intended_biases:
            for repeat_index in range(n_repeats):
                train_indices, test_indices, effective_bias = (
                    self.split_for_intended_bias(
                        smiless,
                        similarity_matrix,
                        activity_values,
                        intended_bias,
                        repeat_index,
                    )
                )
                yield train_indices, test_indices, effective_bias, intended_bias, repeat_index

    @staticmethod
    def effective_bias_from_question_results(question_results):
        if question_results.size == 0:
            return 0.0
        return float(question_results.mean())

    @staticmethod
    def evaluate_cliff_question(
        test_indices,
        train_indices,
        similarity_matrix,
        activity_values,
        activity_threshold,
        similarity_threshold,
    ):
        if len(test_indices) == 0:
            return np.array([])

        # similarity[i, j] = similarity between test molecule i and train molecule j
        similarity_test_vs_train = similarity_matrix[
            test_indices[:, None], train_indices
        ]

        # activity_diff[i, j] = |activity(test i) - activity(train j)|
        activity_diff_test_vs_train = np.abs(
            activity_values[test_indices][:, None] - activity_values[train_indices]
        )

        is_cliff_edge = (similarity_test_vs_train >= similarity_threshold) & (
            activity_diff_test_vs_train >= activity_threshold
        )

        # A test molecule counts if it has at least one cliff edge to any train molecule.
        test_molecule_has_cliff_partner = is_cliff_edge.any(axis=1)
        return test_molecule_has_cliff_partner.astype(float)

    @staticmethod
    def find_cliff_edges(
        similarity_matrix,
        activity_values,
        similarity_threshold,
        activity_threshold,
    ):
        n = len(activity_values)
        cliff_edges = []

        for i in range(n):
            for j in range(i + 1, n):  # symmetric matrix
                if similarity_matrix[i, j] < similarity_threshold:  # type: ignore
                    continue
                activity_difference = abs(
                    float(activity_values[i]) - float(activity_values[j])
                )
                if activity_difference >= activity_threshold:
                    cliff_edges.append((i, j, activity_difference))

        return cliff_edges

    @staticmethod
    def compute_cliff_degrees(
        cliff_edges,  # these come from before (node idx1, node idx2, activity_difference)
        n_molecules,
    ):

        degrees = np.zeros(n_molecules, dtype=int)
        for mol_a, mol_b, _ in cliff_edges:
            degrees[mol_a] += 1
            degrees[mol_b] += 1
        return degrees

    @staticmethod
    def walk_cliff_edges(
        cliff_edges, cliff_degrees, n_molecules, n_cliff_test_target, rng
    ):  # this is to ensure reproducibility with random selection
        assignment = np.full(
            n_molecules, UNASSIGNED_NODE
        )  # array with length of n_molecules filled with 0s
        n_cliff_test_placed = 0

        for mol_a, mol_b, _ in cliff_edges:
            if (
                n_cliff_test_placed >= n_cliff_test_target
            ):  # Stop condition as explained above
                break

            status_a = assignment[mol_a]
            status_b = assignment[mol_b]

            if status_a == UNASSIGNED_NODE and status_b == UNASSIGNED_NODE:
                # higher cliff-degree molecule goes to train.
                if cliff_degrees[mol_a] > cliff_degrees[mol_b]:
                    train_molecule, test_molecule = mol_a, mol_b
                elif cliff_degrees[mol_b] > cliff_degrees[mol_a]:
                    train_molecule, test_molecule = mol_b, mol_a
                else:
                    # Equal cliff degree: randomly pick
                    if rng.random() < 0.5:
                        train_molecule, test_molecule = mol_a, mol_b
                    else:
                        train_molecule, test_molecule = mol_b, mol_a

                assignment[train_molecule] = TRAIN_NODE
                assignment[test_molecule] = TEST_NODE
                n_cliff_test_placed += 1

            elif status_a == TRAIN_NODE and status_b == UNASSIGNED_NODE:
                # Unassigned partner of a train molecule goes to test.
                assignment[mol_b] = TEST_NODE
                n_cliff_test_placed += 1

            elif status_b == TRAIN_NODE and status_a == UNASSIGNED_NODE:
                # Same as above with roles swapped.
                assignment[mol_a] = TEST_NODE
                n_cliff_test_placed += 1

            elif status_a == TEST_NODE and status_b == UNASSIGNED_NODE:
                # Unassigned partner of a test molecule goes to train.
                assignment[mol_b] = TRAIN_NODE

            elif status_b == TEST_NODE and status_a == UNASSIGNED_NODE:
                # Same as above just swapped
                assignment[mol_a] = TRAIN_NODE

            # If both are already assigned, there is nothing to do for this edge.

        return assignment

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

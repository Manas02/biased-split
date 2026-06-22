import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
from scipy.stats import gaussian_kde

UNASSIGNED_NODE = 0
TRAIN_NODE = 1
TEST_NODE = 2


def visualise_proxy_split(
    proxy_values,
    train_idx,
    test_idx,
    ideal_range_min,
    ideal_range_max,
    effective_bias,
    intended_bias,
    proxy_label="proxy",
    x_range=None,
    filepath=None,
):
    TEST = (0.20, 0.40, 0.60)
    TRAIN = (0.5, 0.5, 0.5)
    IDEAL = (0.65, 0.15, 0.20)

    fig, ax = plt.subplots(figsize=(10, 5))

    train_values = proxy_values[train_idx]
    test_values = proxy_values[test_idx]

    if x_range is None:
        x_min, x_max = float(proxy_values.min()), float(proxy_values.max())
        pad = (x_max - x_min) * 0.05
        x_range = (x_min - pad, x_max + pad)

    x = np.linspace(x_range[0], x_range[1], 500)
    train_kde = gaussian_kde(train_values)
    test_kde = gaussian_kde(test_values)
    train_density = train_kde(x)
    test_density = test_kde(x)

    ax.axvspan(ideal_range_min, ideal_range_max, color=IDEAL, alpha=0.10, linewidth=0)
    ax.fill_between(x, train_density, color=TRAIN, alpha=0.35, linewidth=0)
    ax.fill_between(x, test_density, color=TEST, alpha=0.45, linewidth=0)
    ax.plot(x, train_density, color=TRAIN, linewidth=1)
    ax.plot(x, test_density, color=TEST, linewidth=1)

    ax.set_xlabel(proxy_label)
    ax.set_ylabel("density")
    ax.set_xlim(x_range)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=TRAIN, linewidth=2, label="train"),
        Line2D([0], [0], color=TEST, linewidth=2, label="test"),
        Patch(facecolor=IDEAL, alpha=0.30, label="ideal range"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        ncol=3,
        fontsize=9,
    )

    caption = (
        f"{len(proxy_values)} molecules ({len(train_idx)} train, {len(test_idx)} test). "
        f"ideal range [{ideal_range_min}, {ideal_range_max}]. "
        f"intended bias {intended_bias:.2f}, effective bias {effective_bias:.2f}"
    )
    fig.text(0.5, 0.00, caption, ha="center", fontsize=8, color="0.4")

    if filepath:
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


class ProxySortedSplitter:
    def __init__(
        self, proxy_function, ideal_range_min, ideal_range_max, test_fraction=0.2
    ):
        self.proxy_function = proxy_function
        self.ideal_range_min = ideal_range_min
        self.ideal_range_max = ideal_range_max
        self.test_fraction = test_fraction

    def split_for_intended_bias(
        self, smiless, proxy_values, activity_values, intended_bias, random_seed
    ):
        if not (0.0 <= intended_bias <= 1.0):
            raise ValueError(f"intended_bias must be in [0, 1], got {intended_bias}")

        rng = np.random.default_rng(random_seed)
        n_molecules = len(smiless)
        target_test_size = int(self.test_fraction * n_molecules)
        n_in_range_test_target = int(intended_bias * target_test_size)

        in_range_mask = self.find_in_range_mask(
            proxy_values, self.ideal_range_min, self.ideal_range_max
        )

        assignment = self.walk_in_range_molecules(
            in_range_mask, n_molecules, n_in_range_test_target, rng
        )

        unassigned_indices = np.where(assignment == UNASSIGNED_NODE)[0]
        unassigned_out_of_range_indices = unassigned_indices[
            ~in_range_mask[unassigned_indices]
        ]
        unassigned_in_range_indices = unassigned_indices[
            in_range_mask[unassigned_indices]
        ]

        n_random_fill = target_test_size - int((assignment == TEST_NODE).sum())
        if n_random_fill > 0:
            if len(unassigned_out_of_range_indices) >= n_random_fill:
                random_test_indices = rng.choice(
                    unassigned_out_of_range_indices, size=n_random_fill, replace=False
                )
            else:
                shortfall = n_random_fill - len(unassigned_out_of_range_indices)
                in_range_topup_indices = rng.choice(
                    unassigned_in_range_indices,
                    size=min(shortfall, len(unassigned_in_range_indices)),
                    replace=False,
                )
                random_test_indices = np.concatenate(
                    [unassigned_out_of_range_indices, in_range_topup_indices]
                )
            assignment[random_test_indices] = TEST_NODE

        assignment[assignment == UNASSIGNED_NODE] = TRAIN_NODE

        train_indices = np.where(assignment == TRAIN_NODE)[0]
        test_indices = np.where(assignment == TEST_NODE)[0]

        question_results = self.evaluate_proxy_question(
            test_indices, proxy_values, self.ideal_range_min, self.ideal_range_max
        )
        effective_bias = self.effective_bias_from_question_results(question_results)

        return train_indices, test_indices, effective_bias

    def split(self, smiless, activity_values, intended_biases, n_repeats):
        proxy_values = np.array([self.proxy_function(s) for s in smiless], dtype=float)
        for intended_bias in intended_biases:
            for repeat_index in range(n_repeats):
                train_indices, test_indices, effective_bias = (
                    self.split_for_intended_bias(
                        smiless,
                        proxy_values,
                        activity_values,
                        intended_bias,
                        repeat_index,
                    )
                )
                yield train_indices, test_indices, effective_bias, intended_bias, repeat_index

    @staticmethod
    def find_in_range_mask(proxy_values, ideal_range_min, ideal_range_max):
        return (proxy_values >= ideal_range_min) & (proxy_values <= ideal_range_max)

    @staticmethod
    def walk_in_range_molecules(
        in_range_mask, n_molecules, n_in_range_test_target, rng
    ):
        assignment = np.full(n_molecules, UNASSIGNED_NODE, dtype=np.int8)
        in_range_indices = np.where(in_range_mask)[0]
        if n_in_range_test_target == 0 or len(in_range_indices) == 0:
            return assignment
        n_to_place = min(n_in_range_test_target, len(in_range_indices))
        selected = rng.choice(in_range_indices, size=n_to_place, replace=False)
        assignment[selected] = TEST_NODE
        return assignment

    @staticmethod
    def evaluate_proxy_question(
        test_indices, proxy_values, ideal_range_min, ideal_range_max
    ):
        if len(test_indices) == 0:
            return np.array([], dtype=float)
        test_proxy = proxy_values[test_indices]
        in_range = (test_proxy >= ideal_range_min) & (test_proxy <= ideal_range_max)
        return in_range.astype(float)

    @staticmethod
    def effective_bias_from_question_results(question_results):
        if question_results.size == 0:
            return 0.0
        return float(question_results.mean())

    def visualise_splits(
        self,
        smiless,
        activity_values,
        intended_biases,
        n_repeats,
        output_path,
        duration=500,
        proxy_label="proxy",
    ):
        proxy_values = np.array([self.proxy_function(s) for s in smiless], dtype=float)
        x_min, x_max = float(proxy_values.min()), float(proxy_values.max())
        pad = (x_max - x_min) * 0.05
        x_range = (x_min - pad, x_max + pad)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            frame_index = 0
            for intended_bias in intended_biases:
                for repeat_index in range(n_repeats):
                    train_idx, test_idx, effective_bias = self.split_for_intended_bias(
                        smiless,
                        proxy_values,
                        activity_values,
                        intended_bias,
                        repeat_index,
                    )
                    p = os.path.join(tmpdir, f"frame_{frame_index:04d}.png")
                    visualise_proxy_split(
                        proxy_values,
                        train_idx,
                        test_idx,
                        self.ideal_range_min,
                        self.ideal_range_max,
                        effective_bias,
                        intended_bias,
                        proxy_label=proxy_label,
                        x_range=x_range,
                        filepath=p,
                    )
                    paths.append(p)
                    frame_index += 1
            frames = [Image.open(p) for p in paths]
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )

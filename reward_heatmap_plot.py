import copy
from sklearn.cluster import KMeans
import argparse
import pickle
import random
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from orientation.get_orientation import get_angle
matplotlib.use("Agg")
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import debug_plots
from debug_plots import load_models
from rules import check_rules_one
from subsample_state import get_grid_points
import rules
import orientation.get_orientation
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

number_of_samples = None
number_of_rules = None

def _shift_segment(segment, cx: float, cy: float):
    """
    Return a deep-copied segment whose every (x, y) is expressed
    relative to (cx, cy).  Keeps the original segment intact.
    """
    s = copy.deepcopy(segment)
    for sa in s:               # sa == state-action pair
        sa.position[0] -= cx
        sa.position[1] -= cy
    return s


def segment_to_tensor(segment):
    """
    Convert a single segment to a 1D torch.FloatTensor on CPU.
    Pre-allocates the output and avoids per-segment GPU transfers.
    """
    # number of radar features per timestep
    R = len(segment[0].radars)
    L = len(segment)               # number of timesteps
    step_size = R + 1 + 2          # radars + action + (x,y)
    total_size = L * step_size

    out = torch.empty(total_size, dtype=torch.float32)
    idx = 0
    for state_action_pair in segment:
        # copy radar features
        out[idx:idx+R] = torch.tensor(state_action_pair.radars, dtype=torch.float32)
        idx += R

        # copy action
        out[idx] = float(state_action_pair.action)
        idx += 1


        # copy position (x, y)
        out[idx:idx+2] = torch.tensor(state_action_pair.position, dtype=torch.float32)
        idx += 2

    return out  # stays on CPU


def _process_segment(args: tuple[int, list, int]) -> tuple[int, int]:
    """
    Light-weight worker function.

    Args
    ----
    args : (idx, segment, num_rules)
        idx         – index of `segment` in the original list (stays in parent)
        segment     – the trajectory segment itself
        num_rules   – how many rules to evaluate

    Returns
    -------
    (reward_value, idx)  – small, fixed-size tuple that is cheap to pipe back.
    """
    idx, segment, num_rules = args
    _, reward_value, _ = check_rules_one(segment, num_rules)  # heavy work
    return reward_value, idx


def parallel_map_large(
    func,
    iterable,
    desc: str | None = None,
    chunk_size: int = 10_000,
    total: int | None = None,
):
    """
    Parallel map that keeps one Pool alive, recycles workers after a while,
    and handles Ctrl-C cleanly.

    • spawn context → CUDA-safe
    • maxtasksperchild → prevents memory / semaphore leaks
    • imap_unordered  → streams results, keeps pipes short
    """
    ctx = mp.get_context("spawn")
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            pass  # iterable is a generator

    with ctx.Pool(
        processes=ctx.cpu_count(),
        maxtasksperchild=100,   # recycle workers every 100 tasks
    ) as pool:
        try:
            for result in tqdm(
                pool.imap_unordered(func, iterable, chunksize=chunk_size),
                total=total,
                desc=desc,
            ):
                yield result
        except KeyboardInterrupt:
            pool.terminate()
            raise


def accuracy_per_xy(
    trajectory_segments,
    number_of_rules,
    reward_model_directory=None,
    reward_model=None,
):
    """
    Sequential (single-process) computation of per-(x,y) accuracy.
    """

    # ── 1. Evaluate rules ───────────────────────────────────────────────────
    segments_by_reward: dict[int, list[int]] = {0: [], 1: []}

    print("Evaluating rule set on segments (sequential)...")
    for idx, seg in enumerate(tqdm(trajectory_segments)):
        _, reward_value, _ = check_rules_one(seg, number_of_rules)
        segments_by_reward[reward_value].append(idx)

    print(
        f"Number of segments with reward 0: {len(segments_by_reward[0])}, "
        f"reward 1: {len(segments_by_reward[1])}"
    )

    # ── 2. Build positive / negative pairs and shift coordinates ────────────
    paired_segments_0, paired_segments_1, coordinate_list = [], [], []

    cx, cy = orientation.get_orientation.CIRCLE_CENTER

    for reward_value, idx_list in segments_by_reward.items():
        opposite_idx_list = segments_by_reward[1 ^ reward_value]
        if not opposite_idx_list:
            raise ValueError(f"No segments with reward {1 ^ reward_value}")

        for idx in idx_list:
            seg_a_world = trajectory_segments[idx]
            seg_b_world = trajectory_segments[random.choice(opposite_idx_list)]

            # Deep-copied, shifted versions for the model
            seg_a = _shift_segment(seg_a_world, cx, cy)
            seg_b = _shift_segment(seg_b_world, cx, cy)

            if reward_value == 0:
                paired_segments_0.append(seg_a)
                paired_segments_1.append(seg_b)
            else:
                paired_segments_0.append(seg_b)
                paired_segments_1.append(seg_a)

            coordinate_list.append((seg_a[0].position[0], seg_a[0].position[1]))

    number_of_pairs = len(paired_segments_0)
    assert number_of_pairs == len(paired_segments_1) == len(coordinate_list)

    # ── 3. Convert to tensors ───────────────────────────────────────────────
    print("Converting segments to flat tensors...")
    tensor_list_0 = torch.stack([segment_to_tensor(s) for s in tqdm(paired_segments_0)])
    tensor_list_1 = torch.stack([segment_to_tensor(s) for s in tqdm(paired_segments_1)])

    # ── 4. Inference and per-xy accuracy ────────────────────────────────────
    model = reward_model or load_models([reward_model_directory])[0]
    model = model.to(device).eval()

    batch_size = 8_192
    correctness = np.empty(number_of_pairs, dtype=np.int32)

    with torch.no_grad():
        for i in range(0, number_of_pairs, batch_size):
            j = i + batch_size
            b0 = tensor_list_0[i:j].to(device, non_blocking=True)
            b1 = tensor_list_1[i:j].to(device, non_blocking=True)
            r0 = model(b0).cpu().numpy().ravel()
            r1 = model(b1).cpu().numpy().ravel()
            correctness[i:j] = (r1 > r0).astype(np.int32)

    overall = correctness.mean()
    coords = np.array(coordinate_list)
    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    sums = np.bincount(inv, weights=correctness, minlength=uniq.shape[0])
    cnts = np.bincount(inv, minlength=uniq.shape[0])
    accs = (sums / cnts).tolist()
    xs, ys = uniq[:, 0].tolist(), uniq[:, 1].tolist()

    print(f"Mean per-XY accuracy: {float(np.mean(accs))}")
    print(f"Overall accuracy:      {overall}")

    return xs, ys, accs


def get_cluster_centers_and_angles(x, y, n_clusters=20):

    points = np.column_stack((x, y))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(points)

    centers = kmeans.cluster_centers_

    centers_and_angles = []
    for center in centers:
        angle = get_angle(center[0], center[1])
        if angle is not None:
            centers_and_angles.append((center[0], center[1], angle))
        else:
            print(f"Invalid angle for center {center}")

    return centers_and_angles

def plot_reward_heatmap(
    samples,
    reward_model_directory=None,
    number_of_rules=1,
    reward_model=None,
    figure_path="figures/",
    arrows=False,
):
    assert reward_model_directory or reward_model, (
        "Must provide either reward_model_directory or reward_model"
    )
    x, y, accuracy = accuracy_per_xy(
        samples, number_of_rules, reward_model_directory, reward_model
    )

    print("Plotting...")

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x, y, c=accuracy, cmap="viridis", alpha=0.7)
    color_bar = plt.colorbar(scatter)
    color_bar.set_label("Accuracy")

    # Flip Y-axis so that pygame-style y (down-positive) plots correctly
    ax.invert_yaxis()

    if arrows:
        centers_and_angles = get_cluster_centers_and_angles(x, y)
        arrow_length = 0.05 * (max(x) - min(x))
        for center_x, center_y, angle in centers_and_angles:
            dx = arrow_length * np.cos(np.radians(angle))
            dy = arrow_length * np.sin(np.radians(angle))
            arrow = FancyArrowPatch(
                (center_x, center_y),
                (center_x + dx, center_y + dy),
                arrowstyle='-|>',
                mutation_scale=15,
                color='red',
                linewidth=1.5,
                zorder=10
            )
            ax.add_patch(arrow)
        ax.plot([], [], color='red', marker='>', linestyle='-',
                label='Car Orientation', markersize=8)
    ax.legend()

    if number_of_samples and number_of_rules:
        plt.title(
            f"Reward Heatmap ({number_of_samples} samples, {number_of_rules} rules)"
        )
    else:
        plt.title("Reward Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(min(x), max(x))

    if number_of_samples and number_of_rules:
        plt.savefig(
            f"{figure_path}reward_heatmap_{number_of_samples}_t_{number_of_rules}_r.png", dpi=600
        )
        print(
            f"Saved to {figure_path}reward_heatmap_{number_of_samples}_samples_{number_of_rules}_rules.png"
        )
    else:
        plt.savefig(f"{figure_path}reward_heatmap.png", dpi=600)
        print(f"Saved to {figure_path}reward_heatmap.png")
    plt.close()


def get_samples(hyperparameter_path="best_params.yaml", sample_pkl=None):
    if not sample_pkl:
        samples = get_grid_points()
    else:
        with open(sample_pkl, "rb") as f:
            samples = pickle.load(f)
    with open(hyperparameter_path, "r") as file:
        data = yaml.safe_load(file)
        debug_plots.hidden_size = data["hidden_size"]

    print("Flattening samples...")
    flattened_samples = [item for sublist in samples for item in sublist]
    return flattened_samples


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )
    parse.add_argument(
        "-m",
        "--model",
        type=str,
        nargs=1,
        help="Path to model weights",
    )
    parse.add_argument(
        "-r",
        "--rules",
        type=int,
        nargs=1,
        help="Number of rules",
    )
    parse.add_argument(
        "-s",
        "--samples",
        type=int,
        nargs=1,
        help="Number of samples",
    )
    parse.add_argument(
        "-a",
        "--arrows",
        action="store_true",
        help="Whether to show arrows in the plot",
    )
    args = parse.parse_args()
    if args.samples:
        number_of_samples = args.samples[0]
    if args.rules:
        number_of_rules = args.rules[0]
    rules.NUMBER_OF_RULES = number_of_rules
    rules.RULES_INCLUDED = list(range(1, number_of_rules + 1))
    print(
        f"Using {rules.NUMBER_OF_RULES} samples and {rules.RULES_INCLUDED} rules."
    )
    start = time.time()
    plot_reward_heatmap(
        get_samples(sample_pkl="grid_points.pkl"), args.model[0], args.rules[0], arrows=args.arrows
    )
    end = time.time()
    print(f"Finished in {end - start} seconds.")

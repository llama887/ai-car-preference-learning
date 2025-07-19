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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

number_of_samples = None
number_of_rules = None


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


def _process_segment(args):
    segment, number_of_rules = args
    s = copy.deepcopy(segment)
    _, reward_value, _ = check_rules_one(s, number_of_rules)
    cx, cy = orientation.get_orientation.CIRCLE_CENTER
    s[0].position[0] -= cx
    s[0].position[1] -= cy
    s[1].position[0] -= cx
    s[1].position[1] -= cy
    return reward_value, s

def parallel_map_large(
    func, iterable, desc=None, chunk_size=10000, total=None
):
    """
    Parallel map with chunking for large data.
    Yields results in order.
    """
    if total is None:
        try:
            total = len(iterable)
        except Exception:
            total = None
    for i in tqdm(range(0, len(iterable), chunk_size), desc=desc):
        chunk = iterable[i:i+chunk_size]
        with Pool(int(cpu_count()*1.5)) as pool:
            yield from pool.map(func, chunk)

def accuracy_per_xy(
    trajectory_segments,
    number_of_rules,
    reward_model_directory=None,
    reward_model=None
):

    print("Splitting segments by reward and xy...")
    # Chunked parallel processing
    results = list(
        parallel_map_large(
            _process_segment,
            [(seg, number_of_rules) for seg in trajectory_segments],
            desc="Processing segments",
            chunk_size=100000,
            total=len(trajectory_segments)
        )
    )

    segments_by_reward = {0: [], 1: []}
    for reward_value, s in results:
        segments_by_reward[reward_value].append(s)

    print(
        f"Number of segments with reward 0: {len(segments_by_reward[0])}, "
        f"reward 1: {len(segments_by_reward[1])}"
    )

    # 2) Build paired lists as before
    paired_segments_0 = []
    paired_segments_1 = []
    coordinate_list = []
    for reward_value, segment_list in segments_by_reward.items():
        opposite_list = segments_by_reward[1 if reward_value == 0 else 0]
        assert opposite_list, f"No segments with reward {1 if reward_value == 0 else 0}"
        for original_segment in segment_list:
            sampled_opposite = random.choice(opposite_list)
            if reward_value == 0:
                paired_segments_0.append(original_segment)
                paired_segments_1.append(sampled_opposite)
            else:
                paired_segments_0.append(sampled_opposite)
                paired_segments_1.append(original_segment)
            x_coord, y_coord = original_segment[0].position
            coordinate_list.append((x_coord, y_coord))

    number_of_pairs = len(paired_segments_0)
    assert number_of_pairs == len(paired_segments_1) == len(coordinate_list)


    # 4) Sequential tensor conversion (original version)
    print("Converting segments to flat tensors...")
    tensor_list_0 = torch.stack([
        segment_to_tensor(seg) for seg in tqdm(paired_segments_0)
    ])
    tensor_list_1 = torch.stack([
        segment_to_tensor(seg) for seg in tqdm(paired_segments_1)
    ])

    # 5) GPU inference + correctness
    model = reward_model or load_models([reward_model_directory])[0]
    model = model.to(device).eval()
    batch_size = 8192
    correctness = np.empty(number_of_pairs, dtype=np.int32)
    with torch.no_grad():
        for i in range(0, number_of_pairs, batch_size):
            j = i + batch_size
            b0 = tensor_list_0[i:j].to(device, non_blocking=True)
            b1 = tensor_list_1[i:j].to(device, non_blocking=True)
            r0 = model(b0).cpu().numpy().ravel()
            r1 = model(b1).cpu().numpy().ravel()
            correctness[i:j] = (r1 > r0).astype(np.int32)

    # 6) Compute accuracies per (x,y)
    overall = correctness.mean()
    coords = np.array(coordinate_list)
    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    sums = np.bincount(inv, weights=correctness, minlength=uniq.shape[0])
    cnts = np.bincount(inv, minlength=uniq.shape[0])
    accs = (sums / cnts).tolist()
    xs, ys = uniq[:,0].tolist(), uniq[:,1].tolist()

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

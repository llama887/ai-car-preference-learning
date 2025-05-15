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
    for sap in segment:
        # copy radar features
        out[idx:idx+R] = torch.tensor(sap.radars, dtype=torch.float32)
        idx += R

        # copy action
        out[idx] = float(sap.action)
        idx += 1

        # copy position (x, y)
        out[idx:idx+2] = torch.tensor(sap.position, dtype=torch.float32)
        idx += 2

    return out  # stays on CPU

def accuracy_per_xy(
    trajectory_segments,
    number_of_rules,
    reward_model_directory=None,
    reward_model=None
):
    print("Splitting segments by reward and xy…")
    segment_infos = []
    reward_groups = {0: [], 1: []}
    for seg in trajectory_segments:
        _, reward, _ = check_rules_one(seg, number_of_rules)
        assert reward in (0, 1)
        x, y = seg[0].position
        segment_infos.append({"segment": seg, "reward": reward, "xy": (x, y)})
        reward_groups[reward].append(seg)

    # Build random opposite-reward pairs (with replacement)
    pairs_by_xy = {}
    for info in segment_infos:
        seg, reward, xy = info["segment"], info["reward"], info["xy"]
        opp_list = reward_groups[1 - reward]
        if not opp_list:
            print(f"  ⚠️ No opposite-reward partner for xy={xy}, reward={reward}; skipping.")
            continue

        partner = random.choice(opp_list)
        seg0, seg1 = (seg, partner) if reward == 0 else (partner, seg)
        pairs_by_xy.setdefault(xy, []).append((seg0, seg1))

    # Drop xy with no pairs
    start = len(pairs_by_xy)
    for xy in list(pairs_by_xy):
        if not pairs_by_xy[xy]:
            del pairs_by_xy[xy]
    kept = len(pairs_by_xy)
    print(f"Kept {kept} xy points (dropped {start - kept} with no valid pairs).")
    if kept == 0:
        return [], [], []

    # Flatten pairs and collect xy list
    xy_list = list(pairs_by_xy)
    all_pairs = [pair for xy in xy_list for pair in pairs_by_xy[xy]]

    # Load or receive model, move it to device
    if reward_model is None:
        model, _ = load_models([reward_model_directory])
    else:
        model = reward_model
    model = model.to(device)
    model.eval()

    batch_size = 4096

    # Convert every segment to a CPU tensor (no .to(device) here)
    tensor_0 = [
        segment_to_tensor(p[0]) for p in tqdm(all_pairs, desc="Converting to tensors 0")
    ]
    tensor_1 = [
        segment_to_tensor(p[1]) for p in tqdm(all_pairs, desc="Converting to tensors 1")
    ]

    # Build DataLoader with parallel workers and pinned memory
    dataset = TensorDataset(torch.stack(tensor_0), torch.stack(tensor_1))
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Score in batches: one GPU transfer per batch
    batch_accs = []
    for b0, b1 in tqdm(loader, desc="Scoring pairs"):
        b0 = b0.to(device, non_blocking=True)
        b1 = b1.to(device, non_blocking=True)
        o0 = model(b0).detach().cpu().numpy()
        o1 = model(b1).detach().cpu().numpy()
        batch_accs.extend((o0 < o1).astype(int))

    # Group back by xy and compute per-xy accuracy
    accuracies = []
    idx = 0
    for xy in xy_list:
        n = len(pairs_by_xy[xy])
        slice_ = batch_accs[idx : idx + n]
        accuracies.append(sum(slice_) / n)
        idx += n

    all_x = [xy[0] for xy in xy_list]
    all_y = [xy[1] for xy in xy_list]
    import statistics
    mean_accuracy = statistics.mean(accuracies)
    print(f"Mean accuracy: {mean_accuracy:.3f}")
    return all_x, all_y, accuracies


def get_cluster_centers_and_angles(x, y, n_clusters=13):
    from sklearn.cluster import KMeans

    points = np.column_stack((x, y))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(points)

    centers = kmeans.cluster_centers_

    centers_and_angles = []
    for center in centers:
        angle = get_angle(center[0], -center[1])
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

    y = list(map(lambda x: -x, y))

    print("Plotting...")

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(x, y, c=accuracy, cmap="viridis", alpha=0.7)
    color_bar = plt.colorbar(scatter)
    color_bar.set_label("Accuracy")

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
    plt.ylim(min(y), max(y))

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
    args = parse.parse_args()
    if args.samples:
        number_of_samples = args.samples[0]
    if args.rules:
        number_of_rules = args.rules[0]

    start = time.time()
    plot_reward_heatmap(
        get_samples(sample_pkl="grid_points.pkl"), args.model[0], args.rules[0]
    )
    end = time.time()
    print(f"Finished in {end - start} seconds.")

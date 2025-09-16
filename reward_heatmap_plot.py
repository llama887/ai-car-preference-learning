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
from typing import List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arrow plotting params 
ARROW_N_CLUSTERS: int = 16          # number of arrow clusters
ARROW_SCALE: float = 10.0            # multiply the averaged dx, dy by this
ARROW_MIN_MAG: float = 1         # minimum |vector| in data units to draw



number_of_samples = None
number_of_rules = None

def _compute_clustered_arrows_from_segments(
    segments: List[List],
    n_arrows: int,
    min_arrow_magnitude: float,
    arrow_scale: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Build arrows directly from raw segments (no binning).
    Each segment contributes:
      start = (x1, y1)
      disp  = (x2 - x1, y2 - y1)
    Then KMeans on starts -> one averaged arrow per cluster.

    Returns (qx, qy, qu, qv) ready for plt.quiver, all in DATA units.
    """
    # Normalize to match the heatmap's coordinate system
    normalized: List[List] = [
        _shift_segment(seg, *orientation.get_orientation.CIRCLE_CENTER) for seg in segments
    ]

    starts_xy: List[Tuple[float, float]] = []
    disps_xy:  List[Tuple[float, float]] = []

    for seg in normalized:
        if len(seg) < 2:
            continue
        x1 = float(seg[0].position[0]); y1 = float(seg[0].position[1])
        x2 = float(seg[1].position[0]); y2 = float(seg[1].position[1])
        starts_xy.append((x1, y1))
        disps_xy.append((x2 - x1, y2 - y1))

    if not starts_xy:
        return [], [], [], []

    pts = np.asarray(starts_xy, dtype=float)
    vec = np.asarray(disps_xy,  dtype=float)

    k = int(min(max(1, n_arrows), len(pts)))
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pts)

    qx: List[float] = []
    qy: List[float] = []
    qu: List[float] = []
    qv: List[float] = []

    for cid in range(k):
        mask = labels == cid
        if not np.any(mask):
            continue
        cluster_pts = pts[mask]
        cluster_vec = vec[mask]

        # Arrow tail at cluster center; arrow is average displacement
        cx, cy = cluster_pts.mean(axis=0).tolist()
        vx, vy = cluster_vec.mean(axis=0).tolist()

        mag = float(np.hypot(vx, vy))
        if mag < min_arrow_magnitude:
            continue

        qx.append(cx)
        qy.append(cy)
        qu.append(vx * arrow_scale)
        qv.append(vy * arrow_scale)

    return qx, qy, qu, qv


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

    return out  



def accuracy_per_xy(
    trajectory_segments,
    number_of_rules,
    reward_model_directory=None,
    reward_model=None,
):
    """
    Computation of per-(x,y) accuracy.
    """
    normalized_segments = [_shift_segment(seg, *orientation.get_orientation.CIRCLE_CENTER) for seg in tqdm(trajectory_segments, desc="Normalizing segments")]

    segments_by_reward: dict[int, list[int]] = {0: [], 1: []}

    for idx, seg in enumerate(tqdm(normalized_segments, desc="Evaluating rule set on segments")):
        _, reward_value, _ = check_rules_one(seg, number_of_rules)
        assert reward_value in (0, 1), f"Unexpected reward value {reward_value}"
        segments_by_reward[reward_value].append(idx)

    print(
        f"Number of segments with reward 0: {len(segments_by_reward[0])}, "
        f"reward 1: {len(segments_by_reward[1])}"
    )

    paired_segments_0, paired_segments_1, coordinate_list = [], [], []

    r0_idx_list = segments_by_reward[0]
    r1_idx_list = segments_by_reward[1]

    for idx in tqdm(r0_idx_list, desc="Pairing segments with reward 0"):
        r0_segment = normalized_segments[idx]
        r1_segment = normalized_segments[random.choice(r1_idx_list)]
        paired_segments_0.append(r0_segment)
        paired_segments_1.append(r1_segment)
        _, r0, _ = check_rules_one(r0_segment, number_of_rules)
        _, r1, _ = check_rules_one(r1_segment, number_of_rules)
        assert r0 == 0 and r1 == 1, f"Expected rewards 0 and 1, got {r0} and {r1}"
        coordinate_list.append((r0_segment[0].position[0], r0_segment[0].position[1]))

    for idx in tqdm(r1_idx_list, desc="Pairing segments with reward 1"):
        r1_segment = normalized_segments[idx]
        r0_segment = normalized_segments[random.choice(r0_idx_list)]
        paired_segments_0.append(r0_segment)
        paired_segments_1.append(r1_segment)
        _, r0, _ = check_rules_one(r0_segment, number_of_rules)
        _, r1, _ = check_rules_one(r1_segment, number_of_rules)
        assert r0 == 0 and r1 == 1, f"Expected rewards 0 and 1, got {r0} and {r1}"
        coordinate_list.append((r1_segment[0].position[0], r1_segment[0].position[1]))

    number_of_pairs = len(paired_segments_0)
    assert number_of_pairs == len(paired_segments_1) == len(coordinate_list)

    tensor_list_0 = torch.stack([segment_to_tensor(s) for s in tqdm(paired_segments_0, desc="Converting segments with reward 0 to tensors")])
    tensor_list_1 = torch.stack([segment_to_tensor(s) for s in tqdm(paired_segments_1, desc="Converting segments with reward 1 to tensors")])

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
        qx, qy, qu, qv = _compute_clustered_arrows_from_segments(
            samples,
            n_arrows=ARROW_N_CLUSTERS,
            min_arrow_magnitude=ARROW_MIN_MAG,
            arrow_scale=ARROW_SCALE,
        )
        if len(qx) > 0:
            plt.quiver(
                qx, qy, qu, qv,
                angles="xy", scale_units="xy", scale=1,
                width=0.004, alpha=0.95,
            )
            ax.plot([], [], color='k', linestyle='-', label='Average motion (clustered)')
    ax.legend()

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

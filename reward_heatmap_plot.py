import pickle
import random
import time

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import debug_plots
from debug_plots import load_models
from rules import check_rules_one
from subsample_state import get_grid_points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def segment_to_tensor(segment):
    flatten = []
    for state_action_pair in segment:
        flatten.extend([radar for radar in state_action_pair.radars])
        flatten.append(state_action_pair.action)
        flatten.extend(state_action_pair.position)
    return torch.tensor(flatten, dtype=torch.float32).to(device)


def accuracy_per_xy(
    trajectory_segments, number_of_rules, reward_model_directory=None, reward_model=None
):
    # Split by x and y
    print("Splitting by x and y...")
    count1 = 0
    count2 = 0
    count_diff = 0
    diff_xys = []
    xy_dict = {}
    for segment in trajectory_segments:
        _, reward, _ = check_rules_one(segment, number_of_rules)
        assert reward in [0, 1]
        x = segment[0].position[0]
        y = segment[0].position[1]
        if (x, y) not in xy_dict:
            xy_dict[(x, y)] = [[], []]
        xy_dict[(x, y)][reward].append(segment)
        if (
            len(xy_dict[(x, y)][reward]) == 1
            and len(xy_dict[(x, y)][int(not bool(reward))]) != 0
        ):
            count_diff += 1
            diff_xys.append((x, y))
        if reward == 1:
            count1 += 1
        else:
            count2 += 1
    print(
        f"Out of {len(xy_dict)} xy points, {count_diff} have segments that satisfy both rewards."
    )
    print(f"Found {count1} 1 and {count2} 0 examples.")
    # make pairs
    print("Making pairs...")
    paired_dict = {}
    for xy in xy_dict:
        number_of_pairs = min(len(xy_dict[xy][0]), len(xy_dict[xy][1]))
        xy_dict[xy][0] = xy_dict[xy][0][:number_of_pairs]
        xy_dict[xy][1] = xy_dict[xy][1][:number_of_pairs]
        paired_dict[xy] = [
            (xy_dict[xy][0][i], xy_dict[xy][1][i]) for i in range(number_of_pairs)
        ]

    # remove xy with no pairs
    # xy can have no pairs if everything satisfies the same number of rules
    print("Removing xy with no pairs...")
    start_count = len(xy_dict)
    for xy in list(xy_dict.keys()):
        random.shuffle(xy_dict[xy][0])
        random.shuffle(xy_dict[xy][1])
        if len(xy_dict[xy][0]) == 0 or len(xy_dict[xy][1]) == 0:
            del xy_dict[xy]
    end_count = len(xy_dict)
    print(f"Reduced from {start_count} to {end_count} xy pairs.")

    x = []
    y = []
    accuracy = []

    # Evaluate accuracy with batching
    print("Evaluating accuracy with batching...")
    if not reward_model:
        model, _ = load_models([reward_model_directory])
    else:
        model = reward_model
    batch_size = 1024

    # Prepare data for batching
    all_x, all_y, all_pairs = [], [], []

    for xy in paired_dict:
        all_x.append(xy[0])
        all_y.append(xy[1])
        all_pairs.extend(paired_dict[xy])

    # Convert pairs to tensors
    tensor_pairs_0 = [
        segment_to_tensor(pair[0])
        for pair in tqdm(all_pairs, desc="Converting to tensors 1")
    ]
    tensor_pairs_1 = [
        segment_to_tensor(pair[1])
        for pair in tqdm(all_pairs, desc="Converting to tensors 2")
    ]

    # Create dataset and dataloader for batching
    dataset = TensorDataset(torch.stack(tensor_pairs_0), torch.stack(tensor_pairs_1))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Run batched inference
    batch_accuracies = []
    for batch_0, batch_1 in tqdm(dataloader, desc="Processing batches"):
        outputs_0 = model(batch_0).detach().cpu().numpy()
        outputs_1 = model(batch_1).detach().cpu().numpy()
        batch_accuracies.extend((outputs_0 < outputs_1).astype(int))

    # Map batch results back to XY pairs
    accuracy = []
    current_index = 0
    for xy in paired_dict:
        num_pairs = len(paired_dict[xy])
        xy_accuracy = batch_accuracies[current_index : current_index + num_pairs]
        accuracy.append(sum(xy_accuracy) / len(xy_accuracy))
        current_index += num_pairs

    return all_x, all_y, accuracy


def plot_reward_heatmap(
    samples,
    reward_model_directory=None,
    number_of_rules=1,
    reward_model=None,
    figure_path="figures/",
):
    assert reward_model_directory or reward_model, (
        "Must provide either reward_model_directory or reward_model"
    )
    print("Getting accuracies...")
    x, y, accuracy = accuracy_per_xy(
        samples, number_of_rules, reward_model_directory, reward_model
    )

    print("Negating all Y values...")
    y = list(map(lambda x: -x, y))

    print("Plotting...")
    plt.scatter(x, y, c=accuracy, cmap="viridis")
    color_bar = plt.colorbar()
    color_bar.set_label("Accuracy", fontsize=12)
    plt.title("Reward Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y))
    plt.savefig(f"{figure_path}reward_heatmap.png", dpi=300)
    print(f"Saved to {figure_path}reward_heatmap.png")


def get_samples(hyperparameter_path="best_params.yaml", sample_pkl=None):
    if not sample_pkl:
        samples = get_grid_points(1000000)
    else:
        with open(sample_pkl, "rb") as f:
            samples = pickle.load(f)
    with open(hyperparameter_path, "r") as file:
        data = yaml.safe_load(file)
        debug_plots.hidden_size = data["hidden_size"]

    print("Flattening samples...")
    flatted_samples = [item for sublist in samples for item in sublist]
    return flatted_samples


if __name__ == "__main__":
    start = time.time()
    plot_reward_heatmap(get_samples(), "models/model_100.pth", 1)
    end = time.time()
    print(f"Finished in {end - start} seconds.")

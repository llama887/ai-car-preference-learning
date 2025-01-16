import time

import matplotlib.pyplot as plt
import yaml

import debug_plots
from agent import prepare_single_trajectory
from debug_plots import load_models
from rules import check_rules_one
from subsample_state import get_grid_points


def accuracy_per_xy(trajectory_segments, number_of_rules, reward_model_directory):
    import ipdb

    # Split by x and y
    print("Splitting by x and y...")
    count1 = 0
    count2 = 0
    xy_dict = {}
    for segment in trajectory_segments:
        _, reward, _ = check_rules_one(segment, number_of_rules)
        x = segment[0].position[0]
        y = segment[0].position[1]
        if (x, y) not in xy_dict:
            xy_dict[(x, y)] = [[], []]
        xy_dict[(x, y)][reward].append(segment)
        if reward == 1:
            count1 += 1
        else:
            count2 += 1
    print(f"Found {count1} 1 and {count2} 0 examples.")
    ipdb.set_trace()
    # make pairs
    print("Making pairs...")
    paired_dict = {}
    for xy in xy_dict:
        ipdb.set_trace()
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
        if len(xy_dict[xy][0]) == 0 or len(xy_dict[xy][1]) == 0:
            del xy_dict[xy]
    end_count = len(xy_dict)
    print(f"Reduced from {start_count} to {end_count} xy pairs.")

    x = []
    y = []
    accuracy = []

    # evaluate accuracy
    print("Evaluating accuracy...")
    model, _ = load_models([reward_model_directory])
    for xy in paired_dict:
        x.append(xy[0])
        y.append(xy[1])
        xy_accuracy = [
            int(
                model(
                    prepare_single_trajectory(
                        pair[0],
                    )
                ).item()
                < model(
                    prepare_single_trajectory(
                        pair[1],
                    )
                ).item()
            )
            for pair in paired_dict[xy]
        ]
        accuracy.append(
            (sum(xy_accuracy) / len(xy_accuracy) if len(xy_accuracy) > 0 else 0)
        )
    return x, y, accuracy


def plot_reward_heatmap(samples, reward_model_directory, number_of_rules):
    print("Getting accuracies...")
    x, y, accuracy = accuracy_per_xy(samples, number_of_rules, reward_model_directory)

    plt.figure()
    plt.imshow(accuracy, cmap="viridis", extent=[x.min(), x.max(), y.min(), y.max()])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Reward Heatmap")
    plt.savefig("reward_heatmap.png")
    plt.close()


if __name__ == "__main__":
    start = time.time()
    samples = get_grid_points(1000000)
    with open("best_params.yaml", "r") as file:
        data = yaml.safe_load(file)
        debug_plots.hidden_size = data["hidden_size"]

    print("Flattening samples...")
    flatted_samples = [item for sublist in samples for item in sublist]
    plot_reward_heatmap(flatted_samples, "models/model_1.pth", 1)
    end = time.time()
    print(f"Finished in {end - start} seconds.")

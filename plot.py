from reward import (
    TrajectoryRewardNet,
    prepare_single_trajectory,
)
import reward

figure_path = reward.figure_path

import pickle
import os
import wandb
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET_SIZE = 4


class RewardNormalizer:
    def __init__(self, rewards):
        self.min_val = min(rewards)
        self.max_val = max(rewards)

        # Avoid division by zero if all rewards are the same
        if self.min_val == self.max_val:
            self.min_val = 0
            self.max_val = 1

        self.normalized_rewards = [
            (reward - self.min_val) / (self.max_val - self.min_val)
            for reward in rewards
        ]

    def normalize(self, value):
        # Normalize a new value using the stored min and max values
        if self.min_val == self.max_val:
            return 0.5

        return (value - self.min_val) / (self.max_val - self.min_val)

    def get_normalized_rewards(self):
        return self.normalized_rewards


def bradley_terry(r1, r2):
    return math.exp(r1) / (math.exp(r1) + math.exp(r2))


def prepare_data(database_path, model_weights=None, net=None, hidden_size=None):
    with open(database_path, "rb") as f:
        trajectories = pickle.load(f)
    if model_weights is not None:
        model = TrajectoryRewardNet(NET_SIZE, hidden_size=int(hidden_size)).to(device)
        model.load_state_dict(torch.load(model_weights))
    elif net is not None:
        model = net
    else:
        raise Exception(
            "prepare_data expects either a path to model weights or the reward network"
        )
    model.eval()
    reward1 = [t[-2] for t in trajectories]
    reward2 = [t[-1] for t in trajectories]
    rewards = reward1 + reward2
    true_reward_normalizer = RewardNormalizer(rewards)
    normalized_true_reward1 = true_reward_normalizer.get_normalized_rewards()[
        : len(reward1)
    ]
    normalized_true_reward2 = true_reward_normalizer.get_normalized_rewards()[
        len(reward1) :
    ]
    trained_reward1 = [model(prepare_single_trajectory(t[0])) for t in trajectories]
    trained_reward2 = [model(prepare_single_trajectory(t[1])) for t in trajectories]
    trained_reward_normalizer = RewardNormalizer(trained_reward1 + trained_reward2)
    normalized_trained_reward1 = trained_reward_normalizer.get_normalized_rewards()[
        : len(trained_reward1)
    ]
    normalized_trained_reward2 = trained_reward_normalizer.get_normalized_rewards()[
        len(trained_reward1) :
    ]

    false_true_bradley_terry = []
    false_trained_bradley_terry = []
    true_bradley_terry = []
    bradley_terry_difference = []
    ordered_trajectories = []

    for _, t in enumerate(
        zip(
            normalized_true_reward1,
            normalized_true_reward2,
            normalized_trained_reward1,
            normalized_trained_reward2,
        )
    ):
        (true_r1, true_r2, trained_r1, trained_r2) = t
        true_preference = true_r1 > true_r2
        trained_preference = trained_r1 > trained_r2
        predicted_bradley_terry = bradley_terry(trained_r1, trained_r2)
        true_bradley_terry = bradley_terry(true_r1, true_r2)
        bradley_terry_difference.append(true_bradley_terry - predicted_bradley_terry)
        ordered_trajectories.extend([[trained_r1, true_r1], [trained_r2, true_r2]])
        if trained_preference != true_preference:
            false_true_bradley_terry.append(true_bradley_terry)
            false_trained_bradley_terry.append(predicted_bradley_terry)

    ordered_trajectories = sorted(ordered_trajectories, key=lambda x: x[0])
    return (
        false_true_bradley_terry,
        false_trained_bradley_terry,
        bradley_terry_difference,
        ordered_trajectories,
    )


def populate_lists(
    true_database,
    trained_database,
    true_agent_distances,
    trained_agent_distances,
    trained_agent_rewards,
    trained_segment_distances,
    trained_segment_rewards,
):
    with open(true_database, "rb") as f:
        true_trajectories = pickle.load(f)
    with open(trained_database, "rb") as f:
        trained_trajectories = pickle.load(f)


# Define a function to plot trajectories
def plot_trajectories(trajectories, title):
    plt.figure(figsize=(10, 6))
    for trajectory in trajectories:
        xs, ys = zip(*trajectory)  # Unpack x and y coordinates
        plt.plot(xs, ys, alpha=0.7)  # Plot the trajectory with markers
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(f"{figure_path}/{title}.png")
    plt.close()


def plot_bradley_terry(data1, title, data2=None):
    if data2 is not None:
        sns.histplot(data1, kde=True, color="b", label="Ground Truth")
        sns.histplot(data2, kde=True, color="r", label="Trained", alpha=0.5)
        plt.legend()
    else:
        sns.histplot(data1, kde=True)
    plt.title(title)
    plt.savefig(f"{figure_path}/{title}.png")
    plt.close()


def plot_trajectory_order(data, title):
    bar_heights = [x[1].cpu().item() if torch.is_tensor(x[1]) else x[1] for x in data]
    trained_reward = [
        x[0].cpu().item() if torch.is_tensor(x[0]) else x[0] for x in data
    ]

    x_indices = range(len(data))
    plt.bar(x_indices, bar_heights, align="center", label="Ground Truth Reward")
    plt.bar(
        x_indices,
        trained_reward,
        align="center",
        alpha=0.2,
        label="Trained Reward",
    )
    plt.title("Sorted Trajectories: Trained vs Ground Truth Reward")
    plt.savefig(f"{figure_path}/{title}.png")
    plt.close()


def dist(traj_segment):
    traj_segment_distance = math.sqrt(
        (traj_segment[1][0] - traj_segment[0][0]) ** 2
        + (traj_segment[1][1] - traj_segment[0][1]) ** 2
    )
    return traj_segment_distance


def handle_plotting(
    true_agent_distances,
    trained_agent_distances,
    trained_agent_rewards,
    trained_segment_distances,
    trained_segment_rewards,
):
    traj_per_generation = len(true_agent_distances[0])
    true_reward_averages = [
        (sum(generation) / traj_per_generation) for generation in true_agent_distances
    ]
    true_reward_maxes = [max(generation) for generation in true_agent_distances]

    trained_reward_averages = [
        (sum(generation) / traj_per_generation)
        for generation in trained_agent_distances
    ]
    trained_agent_reward_averages = [
        (sum(generation) / traj_per_generation) for generation in trained_agent_rewards
    ]
    trained_agent_reward_maxes = [
        max(generation) for generation in trained_agent_rewards
    ]
    trained_reward_maxes = [max(generation) for generation in trained_agent_distances]
    graph_avg_max(
        (true_reward_averages, trained_reward_averages),
        (true_reward_maxes, trained_reward_maxes),
    )
    graph_trained_rewards(trained_agent_reward_averages, trained_agent_reward_maxes)
    graph_death_rates(true_agent_distances, "GT")
    graph_death_rates(trained_agent_distances, "Trained")
    # graph_distance_vs_reward(trained_agent_distances, trained_agent_rewards)
    graph_segment_distance_vs_reward(trained_segment_distances, trained_segment_rewards)


def graph_avg_max(averages, maxes):
    true_reward_averages, trained_reward_averages = averages
    true_reward_maxes, trained_reward_maxes = maxes

    os.makedirs("figures", exist_ok=True)

    x_values = range(len(trained_reward_averages))

    plt.figure()
    plt.plot(x_values, true_reward_averages, label="Ground Truth")
    plt.plot(x_values, trained_reward_averages, label="Trained Reward")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("Ground Truth vs Trained Reward: Average Distance")
    plt.legend()
    plt.savefig("figures/average.png")
    plt.close()

    plt.figure()
    plt.plot(x_values, true_reward_maxes, label="Ground Truth")
    plt.plot(x_values, trained_reward_maxes, label="Trained Reward")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("Ground Truth vs Trained Reward: Max Distance")
    plt.legend()
    plt.savefig("figures/max.png")
    plt.close()

    wandb.log(
        {
            "Wandb Avg Plot": wandb.plot.line_series(
                xs=x_values,
                ys=[true_reward_averages, trained_reward_averages],
                keys=["True Average", "Trained Average"],
                title="Ground Truth vs Trained Reward: Average Distance",
            )
        }
    )
    wandb.log(
        {
            "Wandb Max Plot": wandb.plot.line_series(
                xs=x_values,
                ys=[true_reward_maxes, trained_reward_maxes],
                keys=["True Max", "Trained Max"],
                title="Ground Truth vs Trained Reward: Max Distance",
            )
        }
    )


def graph_trained_rewards(averages, maxes):
    os.makedirs("figures", exist_ok=True)

    x_values = range(len(averages))

    plt.figure()
    plt.plot(x_values, averages, label="Average Rewards Per Gen")
    plt.plot(x_values, maxes, label="Max Reward Per Gen")
    plt.xlabel("Generation")
    plt.ylabel("Reward")
    plt.title("Reward Obtained by Trained Agents")
    plt.legend()
    plt.savefig("figures/agent_rewards.png")
    plt.close()

    wandb.log(
        {
            "Wandb Avg Plot": wandb.plot.line_series(
                xs=x_values,
                ys=[averages, maxes],
                keys=["Average Reward", "Max Reward"],
                title="Reward Obtained by Trained Agents",
            )
        }
    )


def graph_death_rates(distances_per_generation, agent_type):
    fig = plt.figure()
    generation_graph = fig.add_subplot(projection="3d")

    traj_per_generation = len(distances_per_generation[0])
    sorted_distances = [sorted(generation) for generation in distances_per_generation]

    xs = []
    ys = []
    generations = []

    for generation, distances_in_generation in enumerate(sorted_distances):
        percent_alive = []
        distance_travelled = []
        for number_dead, traj_distance in enumerate(distances_in_generation):
            percent_alive.append(
                (traj_per_generation - number_dead - 1) / traj_per_generation * 100
            )
            distance_travelled.append(traj_distance)
        xs.append(distance_travelled)
        ys.append(percent_alive)
        generations.append(generation + 1)
        generation_graph.plot(
            distance_travelled,
            percent_alive,
            zs=generation + 1,
            zdir="y",
        )

    generation_graph.set_xlabel("Distance Travelled")
    generation_graph.set_ylabel("Generation")
    generation_graph.set_zlabel("Percent Alive")
    generation_graph.set_yticks(generations)
    plt.title(f"Survival Rate of {agent_type} Agents vs. Distance")
    plt.savefig(f"figures/survival_{agent_type}.png")
    plt.close()

    wandb.log(
        {
            f"Survival Rate of {agent_type} Agents": wandb.plot.line_series(
                xs,
                ys,
                keys=[f"Generation {gen}" for gen in generations],
                title="Ground Truth vs Trained Reward: Average Distance",
            )
        }
    )


def graph_distance_vs_reward(trained_agent_distances, trained_agent_rewards):
    os.makedirs("figures", exist_ok=True)
    aggregate_trained_distance, aggregate_trained_reward = [], []
    for i in range(len(trained_agent_distances)):
        aggregate_trained_distance.extend(trained_agent_distances[i])
        aggregate_trained_reward.extend(trained_agent_rewards[i])
    plt.figure()
    plt.scatter(
        x=aggregate_trained_distance, y=aggregate_trained_reward, label="Trained Agent"
    )
    plt.xlabel("Distance")
    plt.ylabel("Reward")
    plt.title("Reward vs. Distance Travelled")
    plt.legend()
    plt.savefig("figures/agent_distance_vs_reward.png")
    plt.close()

    wandb.log(
        {
            "Wandb Avg Plot": wandb.plot.line_series(
                xs=aggregate_trained_distance,
                ys=[aggregate_trained_reward],
                keys=["Trained Agent"],
                title="Reward vs. Distance Travelled",
            )
        }
    )


def graph_segment_distance_vs_reward(
    trained_segment_distances, trained_segment_rewards
):
    os.makedirs("figures", exist_ok=True)
    plt.figure()
    plt.scatter(
        x=trained_segment_distances,
        y=trained_segment_rewards,
        label="Trained Agent",
        alpha=0.2,
    )
    plt.xlabel("Distance of Trajectory Segment")
    plt.ylabel("Reward of Trajectory Segment")
    plt.title("Reward vs. Distance Travelled")
    plt.legend()
    plt.savefig("figures/agent_segment_distance_vs_reward.png")
    plt.close()

    zipped_distance_reward = list(
        zip(trained_segment_distances, trained_segment_rewards)
    )
    random.shuffle(zipped_distance_reward)
    if len(zipped_distance_reward) % 2 != 0:
        zipped_distance_reward.pop()
    pairs_of_zips = [
        (
            zipped_distance_reward[i],
            zipped_distance_reward[i + 1],
            zipped_distance_reward[i][0] < zipped_distance_reward[i + 1][0],
        )
        for i in range(0, len(zipped_distance_reward), 2)
        if abs(
            round(zipped_distance_reward[i][0])
            - round(zipped_distance_reward[i + 1][0])
        )
        > 0.5
    ]

    acc = 0
    for pair in pairs_of_zips:
        if (pair[0][1] < pair[1][1] and pair[2]) or (
            pair[0][1] > pair[1][1] and not pair[2]
        ):
            acc += 1
    acc /= len(pairs_of_zips)
    print("ACCURACY", acc)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Generatig Plots for trained model")
    parse.add_argument(
        "-d",
        "--database",
        type=str,
        action="append",
        help="Directory to trajectory database file",
    )
    parse.add_argument(
        "-r",
        "--reward",
        type=str,
        help="Directory to Directory to reward function weights",
    )
    parse.add_argument(
        "-hs",
        "--hidden_size",
        type=int,
        help="Hidden size of the model",
    )
    args = parse.parse_args()
    if args.database:
        database = args.database
        true_database = args.database[0]
        trained_database = args.database[1]
    if args.reward:
        reward = args.reward

    bt, bt_, bt_delta, ordered_trajectories = prepare_data(
        trained_database, reward, hidden_size=592
    )
    plot_bradley_terry(bt, "False Bradley Terry", bt_)
    plot_bradley_terry(bt_delta, "Bradley Terry Difference")
    plot_trajectory_order(ordered_trajectories, "Trajectory Order")

    (
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
    ) = ([], [], [], [], [])

    populate_lists(
        true_database,
        trained_database,
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
    )

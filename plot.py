import reward
from reward import TrajectoryRewardNet, prepare_single_trajectory, scaler

figure_path = reward.figure_path

import argparse
import math
import os
import pickle
import random
import re

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET_SIZE = 4
run_wandb = True


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

    @classmethod
    def from_min_max(cls, min_val, max_val):
        return cls([min_val, max_val])

    def normalize(self, value):
        # Normalize a new value using the stored min and max values
        if self.min_val == self.max_val:
            return 0.5

        return (value - self.min_val) / (self.max_val - self.min_val)

    def get_normalized_rewards(self):
        return self.normalized_rewards


def bradley_terry(r1, r2):
    return math.exp(r2) / (math.exp(r1) + math.exp(r2))


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
    true_min, true_max = float("inf"), float("-inf")
    trained_min, trained_max = float("inf"), float("-inf")
    trajectory_segments = []
    trajectory_threshold = 150
    for i, t in enumerate(trajectories):
        if i > trajectory_threshold:
            break
        for segment in break_into_segments(t[0], single=True) + break_into_segments(
            t[1], single=True
        ):
            trajectory_segments.append(segment)
            true_reward = dist(segment)
            true_min = min(true_min, true_reward)
            true_max = max(true_max, true_reward)
            trained_reward = model(prepare_single_trajectory(segment))
            trained_min = min(trained_min, trained_reward)
            trained_max = max(trained_max, trained_reward)
    true_reward_normalizer = RewardNormalizer.from_min_max(true_min, true_max)
    trained_reward_normalizer = RewardNormalizer.from_min_max(trained_min, trained_max)
    random.shuffle(trajectory_segments)
    if len(trajectory_segments) % 2 != 0:
        trajectory_segments.pop()
    false_true_bradley_terry = []
    false_trained_bradley_terry = []
    true_bradley_terry = []
    bradley_terry_difference = []
    ordered_segements = []
    segment_threshold = 12000
    output_size = (
        segment_threshold
        if len(trajectory_segments) > segment_threshold
        else len(trajectory_segments)
    )
    # print("OUTPUT", output_size)
    for i in range(0, output_size, 2):
        distance_1 = dist(trajectory_segments[i])
        distance_2 = dist(trajectory_segments[i + 1])
        true_r1 = true_reward_normalizer.normalize(distance_1)
        true_r2 = true_reward_normalizer.normalize(distance_2)
        trained_r1 = trained_reward_normalizer.normalize(
            model(prepare_single_trajectory(trajectory_segments[i]))
        )
        trained_r2 = trained_reward_normalizer.normalize(
            model(prepare_single_trajectory(trajectory_segments[i + 1]))
        )
        true_preference = true_r1 > true_r2
        trained_preference = trained_r1 > trained_r2
        predicted_bradley_terry = bradley_terry(trained_r1, trained_r2)
        true_bradley_terry = bradley_terry(true_r1, true_r2)
        bradley_terry_difference.append(true_bradley_terry - predicted_bradley_terry)
        ordered_segements.extend([[trained_r1, true_r1], [trained_r2, true_r2]])
        if trained_preference != true_preference:
            false_true_bradley_terry.append(true_bradley_terry)
            false_trained_bradley_terry.append(predicted_bradley_terry)
    ordered_segements = sorted(ordered_segements, key=lambda x: x[0])
    return (
        false_true_bradley_terry,
        false_trained_bradley_terry,
        bradley_terry_difference,
        ordered_segements,
    )


def break_into_segments(trajectory, single=False):
    trajectory_segments = []
    prev = 0
    curr = 1
    while curr < len(trajectory):
        trajectory_segments.append([trajectory[prev], trajectory[curr]])
        prev += 1
        curr += 1
    return trajectory_segments


def dist(traj_segment):
    traj_segment_distance = math.sqrt(
        (traj_segment[1][0] - traj_segment[0][0]) ** 2
        + (traj_segment[1][1] - traj_segment[0][1]) ** 2
    )
    return traj_segment_distance


def populate_lists(
    true_database,
    trained_database,
    training_database,
    agents_per_generation,
    model_weights=None,
    net=None,
    hidden_size=None,
):
    true_agent_distances = []
    trained_agent_distances = []
    trained_agent_rewards = []
    trained_segment_distances = []
    trained_segment_rewards = []
    training_segment_distances = []
    training_segment_rewards = []

    with open(true_database, "rb") as f:
        true_trajectories = pickle.load(f)
    with open(trained_database, "rb") as f:
        trained_trajectories = pickle.load(f)
    with open(training_database, "rb") as f:
        training_trajectories = pickle.load(f)

    if model_weights is not None:
        model = TrajectoryRewardNet(NET_SIZE, hidden_size=int(hidden_size)).to(device)
        model.load_state_dict(torch.load(model_weights))
    elif net is not None:
        model = net
    else:
        raise Exception(
            "prepare_data expects either a path to model weights or the reward network"
        )

    num_true_trajectories = len(true_trajectories)
    count = 0
    while count < num_true_trajectories:
        gen_true_distances = []
        for i in range(agents_per_generation // 2):
            trajectory = true_trajectories[count]
            gen_true_distances.extend([trajectory[3], trajectory[4]])
            count += 1
        if gen_true_distances:
            true_agent_distances.append(gen_true_distances)

    num_trained_trajectories = len(trained_trajectories)
    count = 0
    while count < num_trained_trajectories:
        gen_trained_distances = []
        gen_trained_rewards = []
        for i in range(agents_per_generation // 2):
            trajectory = trained_trajectories[count]
            gen_trained_distances.extend([trajectory[3], trajectory[4]])
            gen_trained_rewards.extend([trajectory[5], trajectory[6]])
            for segment in break_into_segments(trajectory[0]) + break_into_segments(
                trajectory[1]
            ):
                trained_segment_distances.append(dist(segment))
                trained_segment_rewards.append(
                    model(prepare_single_trajectory(segment)).item()
                )
            count += 1
        if gen_trained_distances:
            trained_agent_distances.append(gen_trained_distances)
        if gen_trained_rewards:
            trained_agent_rewards.append(gen_trained_rewards)
    
    for traj1, traj2, _, dist1, dist2 in training_trajectories:
        training_segment_distances.append(dist1)
        training_segment_distances.append(dist2)
        training_segment_rewards.append(model(prepare_single_trajectory(traj1)).item())
        training_segment_rewards.append(model(prepare_single_trajectory(traj2)).item())

    print("SANITY CHECK...")
    lengths = [i for i in range(101)]
    for l in lengths:
        segment = prepare_single_trajectory([[830, 920], [830, 920 + l]])
        print(f"Segment of distance {l} reward:", model(segment).item())

    return (
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
        training_segment_distances,
        training_segment_rewards,
    )


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
    plt.legend()
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
    training_segment_distances,
    training_segment_rewards,
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
    graph_segment_distance_vs_reward("Agent Segment Distance vs Reward", trained_segment_distances, trained_segment_rewards)
    graph_segment_distance_vs_reward("Training Dataset Distance vs Reward", training_segment_distances, training_segment_rewards)


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

    if run_wandb:
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

    if run_wandb:
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

    if run_wandb:
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
        x=aggregate_trained_distance,
        y=aggregate_trained_reward,
        label="Trained Agent",
    )
    plt.xlabel("Distance")
    plt.ylabel("Reward")
    plt.title("Reward vs. Distance Travelled")
    plt.legend()
    plt.savefig("figures/agent_distance_vs_reward.png")
    plt.close()

    if run_wandb:
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
    title, segment_distances, segment_rewards
):

    zipped_distance_reward = list(
        zip(segment_distances, segment_rewards)
    )
    distTotal = {}
    distCount = {}
    for distance, reward in zipped_distance_reward:
        rounded_distance = round(distance, 1)
        distCount[rounded_distance] = 1 + distCount.get(rounded_distance, 0)
        distTotal[rounded_distance] = reward + distTotal.get(rounded_distance, 0)
    print(title)
    for dist in sorted(distCount.keys()):
        print("DISTANCE:", dist, "| COUNT:", distCount[dist])
    rounded_distances = list(distCount.keys())
    avg_reward_for_distances = []
    for rounded_distance in rounded_distances:
        avg_reward_for_distances.append(
            distTotal[rounded_distance] / distCount[rounded_distance]
        )

    os.makedirs("figures", exist_ok=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(
        x=segment_distances,
        y=segment_rewards,
        label="All Traj Segments",
        c="b",
        alpha=0.2,
    )
    ax1.scatter(
        x=rounded_distances,
        y=avg_reward_for_distances,
        label="Avg Reward per Traj Segment Dist.",
        c="r",
    )
    plt.xlabel("Distance of Trajectory Segment")
    plt.ylabel("Reward of Trajectory Segment")
    plt.title(title)
    plt.legend()
    plt.savefig(f"figures/{title}.png")
    plt.close()

    zipped_distance_reward = list(
        zip(segment_distances, segment_rewards)
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
        # if abs(
        #     round(zipped_distance_reward[i][0])
        #     - round(zipped_distance_reward[i + 1][0])
        # )
        # > 0.5
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
    run_wandb = False
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
        try:

            trained_database = args.database[0]
            if len(database) > 1:
                true_database = args.database[1]
            if len(database) > 2:
                training_database = args.database[2]
        except Exception as e:
            pass
    if args.reward:
        with open("scaler.pkl", "rb") as f:
            reward.scaler = pickle.load(f)
        reward = args.reward
    import time

    # bt, bt_, bt_delta, ordered_trajectories = prepare_data(
    #     trained_database, reward, hidden_size=311
    # )
    # plot_bradley_terry(bt, "False Bradley Terry", bt_)
    # plot_bradley_terry(bt_delta, "Bradley Terry Difference")
    # plot_trajectory_order(ordered_trajectories, "Trajectory Order")

    (
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
        training_segment_distances,
        training_segment_rewards,
    ) = populate_lists(
        true_database,
        trained_database,
        training_database,
        agents_per_generation=20,
        model_weights=reward,
        hidden_size=558,
    )

    handle_plotting(
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
        training_segment_distances,
        training_segment_rewards,
    )

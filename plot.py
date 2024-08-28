import reward
from reward import TrajectoryRewardNet, prepare_single_trajectory, scaler
from agent import StateActionPair

figure_path = reward.figure_path

import argparse
import math
import statistics
import os
import pickle
import random
import re
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET_SIZE = 14
run_wandb = True
epochs = None

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
    position_segment = [traj_segment[0].position, traj_segment[1].position]
    traj_segment_distance = math.sqrt(
        (position_segment[1][0] - position_segment[0][0]) ** 2
        + (position_segment[1][1] - position_segment[0][1]) ** 2
    )
    return traj_segment_distance

def calculate_new_point(point, distance, angle):
    x0, y0 = point
    # Convert angle from degrees to radians
    angle_rad = math.radians(angle)
    # Calculate new coordinates
    x1 = x0 + distance * math.cos(angle_rad)
    y1 = y0 + distance * math.sin(angle_rad)
    return [x1, y1]

def populate_lists(
    true_database,
    trained_database,
    training_database,
    model_info
):
    
    model_weights = model_info["weights"]
    net = model_info["net"]
    hidden_size = model_info["hidden-size"]
    agents_per_generation = model_info["agents-per-generation"]

    true_agent_distances = []
    trained_agent_distances = []
    trained_agent_rewards = []
    trained_segment_distances = []
    trained_segment_rewards = []
    training_segment_distances = []
    training_segment_rewards = []
    training_segment_starts = set()
    training_segment_ends = set()

    replace = False
    if true_database == "blank":
        replace = True
    elif true_database:
        with open(true_database, "rb") as f:
            true_trajectories = pickle.load(f)
            # print(true_trajectories[:10])
    if trained_database:
        with open(trained_database, "rb") as f:
            trained_trajectories = pickle.load(f)
    if training_database:
        with open(training_database, "rb") as f:
            training_trajectories = pickle.load(f)
            # print(training_trajectories[:10])

    if model_weights is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TrajectoryRewardNet(NET_SIZE, hidden_size=int(hidden_size)).to(device)
        model.load_state_dict(torch.load(model_weights, map_location=device))
    elif net is not None:
        model = net
    else:
        raise Exception(
            "prepare_data expects either a path to model weights or the reward network"
        )

    if not replace:
        num_true_trajectories = len(true_trajectories)
        count = 0
        while count < num_true_trajectories:
            gen_true_distances = []
            for _ in range(agents_per_generation // 2):
                trajectory_pair = true_trajectories[count]
                gen_true_distances.extend([trajectory_pair.d1, trajectory_pair.d2])
                count += 1
            if gen_true_distances:
                true_agent_distances.append(gen_true_distances)

    num_trained_trajectories = len(trained_trajectories)
    count = 0
    while count < num_trained_trajectories:
        gen_trained_distances = []
        gen_trained_rewards = []
        for _ in range(agents_per_generation // 2):
            trajectory_pair = trained_trajectories[count]
            gen_trained_distances.extend([trajectory_pair.d1, trajectory_pair.d2])
            gen_trained_rewards.extend([trajectory_pair.r1, trajectory_pair.r2])
            for segment in break_into_segments(trajectory_pair.t1) + break_into_segments(
                trajectory_pair.t2
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

    def round_10_point(position):
        x, y = position
        rounded_x = round(x * 100 / 10) * 10
        rounded_y = round(y * 100 / 10) * 10
        return rounded_x, rounded_y
    if training_database:
        for traj1, traj2, _, dist1, dist2 in training_trajectories:
            training_segment_distances.append(dist1)
            training_segment_distances.append(dist2)
            training_segment_starts.add((round_10_point([traj1[0][5], traj1[0][6]])))
            training_segment_starts.add((round_10_point([traj2[0][5], traj2[0][6]])))
            training_segment_ends.add((round_10_point([traj1[1][5], traj1[1][6]])))
            training_segment_ends.add((round_10_point([traj2[1][5], traj2[1][6]])))
            training_segment_rewards.append(
                model(prepare_single_trajectory(traj1)).item()
            )
            training_segment_rewards.append(
                model(prepare_single_trajectory(traj2)).item()
            )

    

    if replace:
        true_agent_distances = [
            [0 for _ in range(len(trained_agent_distances[0]))]
            for _ in range(len(trained_agent_distances))
        ]

    last_distance = true_agent_distances[-1][-1]
    while len(true_agent_distances) < len(trained_agent_distances):
        true_agent_distances.append([last_distance * agents_per_generation])

    return (
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
        training_segment_distances,
        training_segment_rewards,
        list(training_segment_starts),
        list(training_segment_ends),
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

def handle_plotting(
    model_info,
    true_agent_distances,
    trained_agent_distances,
    trained_agent_rewards,
    trained_segment_distances,
    trained_segment_rewards,
    training_segment_distances,
    training_segment_rewards,
    training_segment_starts,
    training_segment_ends,
):
    model_weights = model_info["weights"]  
    hidden_size = model_info["hidden-size"]
    epochs = model_info["epochs"]
    pairs_learned = model_info["pairs-learned"]

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
    graph_distance_vs_reward(trained_agent_distances, trained_agent_rewards)

    graph_segment_distance_vs_reward(
        "Agent Segment Distance vs Reward",
        trained_segment_distances,
        trained_segment_rewards,
        epochs,
        pairs_learned,
    )

    graph_segment_distance_vs_reward(
        "Training Dataset Distance vs Reward",
        training_segment_distances,
        training_segment_rewards,
    )

    # graph_position_rewards(
    #     training_segment_starts,
    #     "start",
    #     model_weights, 
    #     hidden_size,
    # )
    # graph_position_rewards(
    #     training_segment_ends,
    #     "end",
    #     model_weights, 
    #     hidden_size,
    # )

    graph_variances()


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

    zipped_distance_reward = list(zip(aggregate_trained_distance, aggregate_trained_reward))
    distTotal = {}
    distCount = {}
    for distance, reward in zipped_distance_reward:
        rounded_distance = round(distance, 1)
        distCount[rounded_distance] = 1 + distCount.get(rounded_distance, 0)
        distTotal[rounded_distance] = reward + distTotal.get(rounded_distance, 0)

    rounded_distances = list(distCount.keys())
    rounded_distances = [rounded_distances[i] for i in range(0, len(rounded_distances), 5)]
    avg_reward_for_distances = []
    for rounded_distance in rounded_distances:
        avg_reward_for_distances.append(
            distTotal[rounded_distance] / distCount[rounded_distance]
        )

    os.makedirs("figures", exist_ok=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(
        x=aggregate_trained_distance,
        y=aggregate_trained_reward,
        label="Trained Agent",
        c="b",
        alpha=0.2,
    )
    ax1.scatter(
        x=rounded_distances,
        y=avg_reward_for_distances,
        label="Avg Reward per Trajectory Dist.",
        c="r",
    )
    plt.xlabel("Distance")
    plt.ylabel("Reward")
    plt.title("Total Reward vs. Total Distance Travelled")
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


def graph_segment_distance_vs_reward(title, segment_distances, segment_rewards, epochs=None, pairs_learned=None):
    if not segment_distances or not segment_rewards:
        return

    zipped_distance_reward = list(zip(segment_distances, segment_rewards))
    distRewards = defaultdict(list)
    for distance, reward in zipped_distance_reward:
        rounded_distance = round(distance, 1)
        distRewards[rounded_distance].append(reward)

    print(title)
    sorted_distances = sorted(list(distRewards.keys()))
    for sorted_distance in sorted_distances:
        print("DISTANCE:", sorted_distance, "| COUNT:", len(distRewards[sorted_distance]))

    avg_reward_for_distances = []
    variances = {}
    for rounded_distance in sorted_distances:
        avg_reward_for_distances.append(
            statistics.mean(distRewards[rounded_distance])
        )
        variances[rounded_distance] = statistics.variance(distRewards[rounded_distance])

    if epochs and pairs_learned:
        with open(f"output_data/variances_{epochs}_epochs_{pairs_learned}_pairs.pkl", 'wb') as file:
            pickle.dump(variances, file)

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
        x=sorted_distances,
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

    zipped_distance_reward = list(zip(segment_distances, segment_rewards))
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
    ]
    num_segments = len(pairs_of_zips)

    wrong = []
    acc = 0
    reacc = 0
    for zip1, zip2, label in pairs_of_zips:
        if (
            (zip1[1] < zip2[1] and label)
            or (zip1[1] > zip2[1] and not label)
            or not zip1[0]
            or not zip2[0]
        ):
            acc += 1
        if (
            (zip1[1] < zip2[1] and label)
            or (zip1[1] > zip2[1] and not label)
            or not zip1[0]
            or not zip2[0]
            or abs(zip1[0] - zip2[0]) < 0.01
        ):
            reacc += 1
        else:
            wrong.append((zip1, zip2))
    acc /= num_segments
    reacc /= num_segments
    print("ACCURACY", acc)
    print("ACCURACY W/O SAME DIST PAIRS", reacc)
    print("WRONG:")
    count = 0
    for zip1, zip2 in wrong:
        count += 1
        print(
            f"DISTANCES: {zip1[0]:11.8f}, {zip2[0]:11.8f} | REWARDS: {zip1[1]:11.8f}, {zip2[1]:11.8f}"
        )
        if count > 100:
            break
    print("------------------------------------------------------------------\n")

def graph_position_rewards(positions, type, model_weights, hidden_size):
    title = ""
    if type == "start":
        title = "Segment Starts"
    else:
        title = "Segment Ends"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrajectoryRewardNet(NET_SIZE, hidden_size=int(hidden_size)).to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    xs = []
    ys = []
    rewards = []
    for position in positions:
        xs.append(position[0])
        ys.append(position[1])
    print("POINTS:", len(positions))
    min_rew = float('inf')
    for idx in range(len(positions)):
        point = [xs[idx], ys[idx]]
        # print(point)
        avg_reward = 0
        for i in range(0, 100, 5):
            new_point = calculate_new_point(point, i, random.randint(0, 365))
            new_segment = [
                    point,
                    new_point
                ] if type == "start" else [
                    new_point,
                    point
                ]
            avg_reward += model(prepare_single_trajectory(new_segment)).item() / 100
            min_rew = min(min_rew, avg_reward)
        rewards.append(avg_reward)
    offset = abs(min_rew) + 1
    scaled_rewards = [r + offset for r in rewards]
    print("plotting...")
    plt.scatter(xs, ys, s=scaled_rewards, c=rewards, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Rewards')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.savefig(f"{figure_path}/{title}.png")
    plt.close()

        
def graph_variances():
    variance_data = {}
    variance_files = glob.glob("output_data/variances*")
    for variance_file in variance_files:
        epochs = variance_file.split('_')[2]
        pairs_learned = variance_file.split('_')[4]
        with open(variance_file, "rb") as file:
            variance_data[(epochs, pairs_learned)] = pickle.load(file)

    os.makedirs("figures", exist_ok=True)
    plt.figure()

    for epochs, pairs_learned in variance_data.keys():
        rounded_distances = sorted(variance_data[(epochs, pairs_learned)].keys())
        variances = [variance_data[(epochs, pairs_learned)][rounded_distance] for rounded_distance in rounded_distances]

        plt.scatter(
            x=rounded_distances,
            y=variances,
            label=f"{epochs} Epochs, {pairs_learned} Pairs",
            alpha=0.5,
        )

    plt.xlabel("Distance of Trajectory Segment")
    plt.ylabel("Variance of Reward")
    plt.title("Variances for Different Model Parameters")
    plt.legend()
    plt.savefig(f"figures/variances.png")
    plt.close()


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
        help="Directory to reward function weights",
    )
    parse.add_argument(
        "-hs",
        "--hidden_size",
        type=int,
        help="Hidden size of the model",
    )

    args = parse.parse_args()
    epochs = None
    num_pairs_learned = None

    if args.database:
        database = args.database
        try:
            trained_database = args.database[0]
            true_database = None
            training_database = None
            if len(database) > 1:
                true_database = args.database[1]
            if len(database) > 2:
                training_database = args.database[2]
                num_pairs_learned = training_database.split('_')[1].split('.')[0]


        except Exception as e:
            pass
    epochs = None
    if args.reward:
        # with open("scaler.pkl", "rb") as f:
        #     reward.scaler = pickle.load(f)
        reward = args.reward
        epochs = reward.split('_')[1].split('.')[0]

    # bt, bt_, bt_delta, ordered_trajectories = prepare_data(
    #     trained_database, reward, hidden_size=311
    # )
    # plot_bradley_terry(bt, "False Bradley Terry", bt_)
    # plot_bradley_terry(bt_delta, "Bradley Terry Difference")
    # plot_trajectory_order(ordered_trajectories, "Trajectory Order")

    model_info = {
        "weights" : reward, 
        "net" : None,
        "hidden-size" : 558, 
        "epochs": epochs, 
        "pairs-learned" : num_pairs_learned, 
        "agents-per-generation" : 20
    }

    (
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
        training_segment_distances,
        training_segment_rewards,
        training_segment_starts,
        training_segment_ends,
    ) = populate_lists(
        true_database,
        trained_database,
        training_database,
        model_info,
    )

    handle_plotting(
        model_info,
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
        training_segment_distances,
        training_segment_rewards,
        training_segment_starts,
        training_segment_ends,
    )

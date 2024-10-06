import reward
from reward import TrajectoryRewardNet, prepare_single_trajectory, scaler
from agent import StateActionPair, NUMBER_OF_RULES

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
from rules import check_rules

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET_SIZE = 16
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


def populate_lists(true_database, trained_database, training_database, model_info):

    model_weights = model_info["weights"]
    net = model_info["net"]
    hidden_size = model_info["hidden-size"]
    agents_per_generation = model_info["agents-per-generation"]

    true_agent_expert_segments = []
    trained_agent_expert_segments = []
    trained_agent_rewards = []
    trained_segment_rules_satisifed = []
    trained_segment_rewards = []
    training_segment_rules_satisfied = []
    training_segment_rewards = []

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

    # replace is just a fill in for if you don't have matching ground truth data (not sure when this is actually used at this point LOL)
    if not replace:
        num_true_trajectories = len(true_trajectories)
        count = 0
        while count < num_true_trajectories:
            gen_true_expert_segments = []
            for _ in range(agents_per_generation // 2):
                trajectory_pair = true_trajectories[count]
                gen_true_expert_segments.extend(
                    [trajectory_pair.e1, trajectory_pair.e2]
                )
                count += 1
            if gen_true_expert_segments:
                true_agent_expert_segments.append(gen_true_expert_segments)

    num_trained_trajectories = len(trained_trajectories)
    count = 0
    while count < num_trained_trajectories:
        gen_trained_expert_segments = []
        gen_trained_rewards = []
        for _ in range(agents_per_generation // 2):
            trajectory_pair = trained_trajectories[count]
            gen_trained_expert_segments.extend([trajectory_pair.e1, trajectory_pair.e2])
            gen_trained_rewards.extend([trajectory_pair.r1, trajectory_pair.r2])
            for segment in break_into_segments(
                trajectory_pair.t1
            ) + break_into_segments(trajectory_pair.t2):
                trained_segment_rules_satisifed.append(
                    check_rules(
                        segment,
                        NUMBER_OF_RULES,
                    )[0]
                )
                trained_segment_rewards.append(
                    model(prepare_single_trajectory(segment)).item()
                )
            count += 1
        if gen_trained_expert_segments:
            trained_agent_expert_segments.append(gen_trained_expert_segments)
        if gen_trained_rewards:
            trained_agent_rewards.append(gen_trained_rewards)

    if training_database:
        for traj1, traj2, _, dist1, dist2 in training_trajectories:
            training_segment_rules_satisfied.append(dist1)
            training_segment_rules_satisfied.append(dist2)
            training_segment_rewards.append(
                model(prepare_single_trajectory(traj1)).item()
            )
            training_segment_rewards.append(
                model(prepare_single_trajectory(traj2)).item()
            )

    if replace:
        true_agent_expert_segments = [
            [0 for _ in range(len(trained_agent_expert_segments[0]))]
            for _ in range(len(trained_agent_expert_segments))
        ]

    last_distance = true_agent_expert_segments[-1][-1]
    while len(true_agent_expert_segments) < len(trained_agent_expert_segments):
        true_agent_expert_segments.append([last_distance * agents_per_generation])

    return (
        true_agent_expert_segments,
        trained_agent_expert_segments,
        trained_agent_rewards,
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        training_segment_rules_satisfied,
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


def handle_plotting(
    model_info,
    true_agent_expert_segments,
    trained_agent_expert_segments,
    trained_agent_rewards,
    trained_segment_rules_satisifed,
    trained_segment_rewards,
    training_segment_rules_satisfied,
    training_segment_rewards,
):
    model_weights = model_info["weights"]
    hidden_size = model_info["hidden-size"]
    epochs = model_info["epochs"]
    pairs_learned = model_info["pairs-learned"]

    agents_per_generation = len(true_agent_expert_segments[0])
    # Avg/max number of expert segments per trajectory for each agent over generations
    trueRF_average_expert_segments = [
        (sum(generation) / agents_per_generation)
        for generation in true_agent_expert_segments
    ]
    trueRF_max_expert_segments = [
        max(generation) for generation in true_agent_expert_segments
    ]

    # Avg/max number of expert segments per trajectory for each agent over generations
    trainedRF_average_expert_segments = [
        (sum(generation) / agents_per_generation)
        for generation in trained_agent_expert_segments
    ]
    trainedRF_max_expert_segments = [
        max(generation) for generation in trained_agent_expert_segments
    ]

    # Avg/max reward obtained by agents over generations
    trained_agent_reward_averages = [
        (sum(generation) / agents_per_generation)
        for generation in trained_agent_rewards
    ]
    trained_agent_reward_maxes = [
        max(generation) for generation in trained_agent_rewards
    ]

    graph_expert_segments_over_generations(
        (trueRF_average_expert_segments, trainedRF_average_expert_segments),
        (trueRF_max_expert_segments, trainedRF_max_expert_segments),
    )
    graph_trained_agent_performance(
        trained_agent_reward_averages, trained_agent_reward_maxes
    )

    graph_segment_distance_vs_reward(
        "Agent Segment Rules Satisfied vs Reward",
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        epochs,
        pairs_learned,
    )

    graph_segment_distance_vs_reward(
        "Training Dataset Rules Satisfied vs Reward",
        training_segment_rules_satisfied,
        training_segment_rewards,
    )

    graph_variances()


def graph_expert_segments_over_generations(averages, maxes):
    trueRF_average_expert_segments, trainedRF_average_expert_segments = averages
    trueRF_max_expert_segments, trainedRF_max_expert_segments = maxes

    os.makedirs("figures", exist_ok=True)

    x_values = range(len(trainedRF_average_expert_segments))

    plt.figure()
    plt.plot(x_values, trueRF_average_expert_segments, label="Ground Truth")
    plt.plot(x_values, trainedRF_average_expert_segments, label="Trained Reward")
    plt.xlabel("Generation")
    plt.ylabel("Number of Expert Trajectories")
    plt.title("Ground Truth vs Trained Reward: Average Number of Expert Segments")
    plt.legend()
    plt.savefig("figures/average.png")
    plt.close()

    plt.figure()
    plt.plot(x_values, trueRF_max_expert_segments, label="Ground Truth")
    plt.plot(x_values, trainedRF_max_expert_segments, label="Trained Reward")
    plt.xlabel("Generation")
    plt.ylabel("Number of Expert Trajectories")
    plt.title("Ground Truth vs Trained Reward: Max Number of Expert Segments")
    plt.legend()
    plt.savefig("figures/max.png")
    plt.close()


def graph_trained_agent_performance(averages, maxes):
    os.makedirs("figures", exist_ok=True)

    x_values = range(len(averages))

    plt.figure()
    plt.plot(x_values, averages, label="Average Fitness Per Gen")
    plt.plot(x_values, maxes, label="Max Fitness Per Gen")
    plt.xlabel("Generation")
    plt.ylabel("Reward")
    plt.title("Reward Obtained by Trained Agents")
    plt.legend()
    plt.savefig("figures/agent_rewards.png")
    plt.close()


# i did a booboo, this can maybe be revived later, its wrong atm

# def graph_rules_satisfied_vs_reward(
#     trained_segment_rules_satisfied, trained_agent_rewards
# ):
#     os.makedirs("figures", exist_ok=True)

#     rewards_for_rules = [[] for _ in range(NUMBER_OF_RULES + 1)]
#     aggregate_trained_rules_satisfied, aggregate_trained_reward = [], []

#     for i in range(len(trained_segment_rules_satisfied)):
#         aggregate_trained_rules_satisfied.extend(trained_segment_rules_satisfied[i])
#         aggregate_trained_reward.extend(trained_agent_rewards[i])

#     for i in range(len(aggregate_trained_rules_satisfied)):
#         num_rules, reward = (
#             trained_segment_rules_satisfied[i],
#             aggregate_trained_reward[i],
#         )
#         rewards_for_rules[num_rules].append(reward)

#     segment_rules_achieved = [i for i in range(NUMBER_OF_RULES + 1)]
#     avg_reward_for_rules_satisfied = []
#     for i in range(NUMBER_OF_RULES + 1):
#         if rewards_for_rules[i]:
#             avg_reward_for_rules_satisfied.append(
#                 sum(rewards_for_rules[i]) / len(rewards_for_rules[i])
#             )
#         else:
#             segment_rules_achieved.remove(i)

#     os.makedirs("figures", exist_ok=True)
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.scatter(
#         x=aggregate_trained_rules_satisfied,
#         y=aggregate_trained_reward,
#         label="Trained Agent",
#         c="b",
#         alpha=0.2,
#     )
#     ax1.scatter(
#         x=segment_rules_achieved,
#         y=avg_reward_for_rules_satisfied,
#         label="Avg Reward per Trajectory Dist.",
#         c="r",
#     )
#     plt.xlabel("Distance")
#     plt.ylabel("Reward")
#     plt.title("Total Reward vs. Total Distance Travelled")
#     plt.legend()
#     plt.savefig("figures/agent_distance_vs_reward.png")
#     plt.close()


def graph_segment_distance_vs_reward(
    title, segment_rules_satisfied, segment_rewards, epochs=None, pairs_learned=None
):
    if not segment_rules_satisfied or not segment_rewards:
        return

    if len(segment_rules_satisfied) != len(segment_rewards):
        print(
            "Graphing Segment Distance vs. Reward Error: Sizes of input lists do not match!"
        )
        return

    rewards_for_rules = [[] for _ in range(NUMBER_OF_RULES + 1)]

    for i in range(len(segment_rules_satisfied)):
        num_rules, reward = (
            segment_rules_satisfied[i],
            segment_rewards[i],
        )
        rewards_for_rules[num_rules].append(reward)

    print(title)
    for i in range(NUMBER_OF_RULES + 1):
        print("Rules Satisfied:", i, "| COUNT:", len(rewards_for_rules[i]))

    avg_reward_for_rules = []
    variances = []
    variance_dict = {}

    for i in range(NUMBER_OF_RULES + 1):
        if len(rewards_for_rules[i]) > 0:
            avg_reward_for_rules.append(statistics.mean(rewards_for_rules[i]))
        if len(rewards_for_rules[i]) > 1:
            variances.append(statistics.variance(rewards_for_rules[i]))
            variance_dict[i] = statistics.variance(rewards_for_rules[i])

    if epochs and pairs_learned:
        with open(
            f"output_data/variances_{epochs}_epochs_{pairs_learned}_pairs.pkl", "wb"
        ) as file:
            pickle.dump(variance_dict, file)

    os.makedirs("figures", exist_ok=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xticks([i for i in range(NUMBER_OF_RULES + 1)])
    ax1.scatter(
        x=segment_rules_satisfied,
        y=segment_rewards,
        label="All Traj Segments",
        c="b",
        alpha=0.2,
    )
    ax1.scatter(
        x=[i for i in range(NUMBER_OF_RULES + 1) if len(rewards_for_rules[i]) > 0],
        y=avg_reward_for_rules,
        label="Avg Reward per Traj Segment Dist.",
        c="r",
    )
    plt.xlabel("Rules Satisfied")
    plt.ylabel("Reward of Trajectory Segment")
    plt.title(title)
    plt.legend()
    plt.savefig(f"figures/{title}.png")
    plt.close()

    zipped_distance_reward = list(zip(segment_rules_satisfied, segment_rewards))
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
    print("ACCURACY W/O SAME REWARD PAIRS", reacc)
    print("WRONG:")
    # count = 0
    # for zip1, zip2 in wrong:
    #     count += 1
    #     print(
    #         f"DISTANCES: {zip1[0]:11.8f}, {zip2[0]:11.8f} | REWARDS: {zip1[1]:11.8f}, {zip2[1]:11.8f}"
    #     )
    #     if count > 100:
    #         break
    # print("------------------------------------------------------------------\n")


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
    min_rew = float("inf")
    for idx in range(len(positions)):
        point = [xs[idx], ys[idx]]
        # print(point)
        avg_reward = 0
        for i in range(0, 100, 5):
            new_point = calculate_new_point(point, i, random.randint(0, 365))
            new_segment = [point, new_point] if type == "start" else [new_point, point]
            avg_reward += model(prepare_single_trajectory(new_segment)).item() / 100
            min_rew = min(min_rew, avg_reward)
        rewards.append(avg_reward)
    offset = abs(min_rew) + 1
    scaled_rewards = [r + offset for r in rewards]
    print("plotting...")
    plt.scatter(xs, ys, s=scaled_rewards, c=rewards, cmap="viridis", alpha=0.6)
    plt.colorbar(label="Rewards")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.savefig(f"{figure_path}/{title}.png")
    plt.close()


def graph_variances():
    variance_data = {}
    variance_files = glob.glob("output_data/variances*")
    for variance_file in variance_files:
        epochs = variance_file.split("_")[2]
        pairs_learned = variance_file.split("_")[4]
        with open(variance_file, "rb") as file:
            variance_data[(epochs, pairs_learned)] = pickle.load(file)

    os.makedirs("figures", exist_ok=True)
    plt.figure()

    for epochs, pairs_learned in variance_data.keys():
        rule_numbers = sorted(variance_data[(epochs, pairs_learned)].keys())
        variances = [variance_data[(epochs, pairs_learned)][i] for i in rule_numbers]

        plt.scatter(
            x=rule_numbers,
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
                num_pairs_learned = training_database.split("_")[1].split(".")[0]

        except Exception as e:
            pass
    epochs = None
    if args.reward:
        # with open("scaler.pkl", "rb") as f:
        #     reward.scaler = pickle.load(f)
        reward = args.reward
        epochs = reward.split("_")[1].split(".")[0]

    # bt, bt_, bt_delta, ordered_trajectories = prepare_data(
    #     trained_database, reward, hidden_size=311
    # )
    # plot_bradley_terry(bt, "False Bradley Terry", bt_)
    # plot_bradley_terry(bt_delta, "Bradley Terry Difference")
    # plot_trajectory_order(ordered_trajectories, "Trajectory Order")

    model_info = {
        "weights": reward,
        "net": None,
        "hidden-size": 558,
        "epochs": epochs,
        "pairs-learned": num_pairs_learned,
        "agents-per-generation": 20,
    }

    (
        true_agent_expert_segments,
        trained_agent_expert_segments,
        trained_agent_rewards,
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        training_segment_rules_satisfied,
        training_segment_rewards,
    ) = populate_lists(
        true_database,
        trained_database,
        training_database,
        model_info,
    )

    handle_plotting(
        model_info,
        true_agent_expert_segments,
        trained_agent_expert_segments,
        trained_agent_rewards,
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        training_segment_rules_satisfied,
        training_segment_rewards,
    )

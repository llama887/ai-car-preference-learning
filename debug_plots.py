import argparse
import glob
import math
import os
import pickle
import random
import statistics
from collections import Counter, defaultdict, deque

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
import torch
import yaml

import agent
import reward
import rules
from reward import Ensemble, TrajectoryRewardNet, prepare_single_trajectory

AGENTS_PER_GENERATION = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET_SIZE = 16
run_wandb = True
epochs = None

model = None


# Todo:
class TrajectoryData:
    # Dimensions [num_generations][num_agents_per_generation]
    def __init__(self):
        self.gt_rewards = [[]]
        self.trained_rewards = [[]]


class SegmentData:
    # Dimensions [total number of segments (if everything is correct 450 * num_agents_per_generation * num_generations)]
    def __init__(self):
        self.rules_satisfied = []
        self.trained_rewards = []
        self.distances = []


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


def break_into_segments(trajectory):
    trajectory_segments = []
    current_segment = deque(trajectory[: agent.train_trajectory_length + 1])
    for i in range(agent.train_trajectory_length + 1, len(trajectory)):
        current_segment.popleft()
        current_segment.append(trajectory[i])
        trajectory_segments.append(list(current_segment))
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
    global model
    if model_info["net"]:
        model = model_info["net"]
    elif model_info["ensemble"]:
        model = model_info["ensemble"]
    else:
        raise Exception("expecting either a net or ensemble")

    hidden_size = model_info["hidden-size"]
    agents_per_generation = model_info["agents-per-generation"]

    true_agent_satisfaction_segments = []
    true_agent_rewards = []

    trained_agent_satisfaction_segments = []
    trained_agent_rewards = []

    trained_segment_rules_satisifed = []
    trained_segment_rewards = []
    trained_segment_distances = []

    training_segment_rules_satisfied = []
    training_segment_rewards = []
    training_segment_distances = []

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

    # replace is just a fill in for if you don't have matching ground truth data (not sure when this is actually used at this point LOL)
    if not replace:
        num_true_trajectories = len(true_trajectories)
        count = 0
        while count < num_true_trajectories:
            gen_true_satisfaction_segments = []
            gen_true_rewards = []
            for _ in range(agents_per_generation):
                trajectory = true_trajectories[count]
                gen_true_satisfaction_segments.append(
                    trajectory.num_satisfaction_segments
                )
                gen_true_rewards.append(
                    sum(
                        [
                            model(
                                prepare_single_trajectory(
                                    segment, agent.train_trajectory_length + 1
                                )
                            ).item()
                            for segment in break_into_segments(trajectory.traj)
                        ]
                    )
                )
                count += 1
            if gen_true_satisfaction_segments:
                true_agent_satisfaction_segments.append(gen_true_satisfaction_segments)
            if gen_true_rewards:
                true_agent_rewards.append(gen_true_rewards)

    num_trained_trajectories = len(trained_trajectories)
    count = 0
    while count < num_trained_trajectories:
        gen_trained_satisfaction_segments = []
        gen_trained_rewards = []
        for _ in range(agents_per_generation):
            trajectory = trained_trajectories[count]
            gen_trained_satisfaction_segments.append(
                trajectory.num_satisfaction_segments
            )
            gen_trained_rewards.append(trajectory.total_reward)
            for segment in break_into_segments(trajectory.traj):
                trained_segment_rules_satisifed.append(
                    rules.check_rules_one(
                        segment,
                        rules.NUMBER_OF_RULES,
                    )[0]
                )
                trained_segment_rewards.append(
                    model(
                        prepare_single_trajectory(
                            segment, agent.train_trajectory_length + 1
                        )
                    ).item()
                )
                trained_segment_distances.append(dist(segment))
            count += 1
        if gen_trained_satisfaction_segments:
            trained_agent_satisfaction_segments.append(
                gen_trained_satisfaction_segments
            )
        if gen_trained_rewards:
            trained_agent_rewards.append(gen_trained_rewards)

    if training_database:
        for segment1, segment2, _, reward1, reward2 in training_trajectories:
            training_segment_rules_satisfied.append(
                rules.check_rules_one(
                    segment1,
                    rules.NUMBER_OF_RULES,
                )[0]
            )
            training_segment_rules_satisfied.append(
                rules.check_rules_one(
                    segment2,
                    rules.NUMBER_OF_RULES,
                )[0]
            )
            training_segment_rewards.append(
                model(
                    prepare_single_trajectory(
                        segment1, agent.train_trajectory_length + 1
                    )
                ).item()
            )
            training_segment_rewards.append(
                model(
                    prepare_single_trajectory(
                        segment2, agent.train_trajectory_length + 1
                    )
                ).item()
            )
            training_segment_distances.append(dist(segment1))
            training_segment_distances.append(dist(segment2))

    if replace:
        true_agent_satisfaction_segments = [
            [0 for _ in range(len(trained_agent_satisfaction_segments[0]))]
            for _ in range(len(trained_agent_satisfaction_segments))
        ]

    last_distance = true_agent_satisfaction_segments[-1][-1]
    while len(true_agent_satisfaction_segments) < len(
        trained_agent_satisfaction_segments
    ):
        true_agent_satisfaction_segments.append([last_distance * agents_per_generation])

    return (
        true_agent_satisfaction_segments,
        true_agent_rewards,
        trained_agent_satisfaction_segments,
        trained_agent_rewards,
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        trained_segment_distances,
        training_segment_rules_satisfied,
        training_segment_rewards,
        training_segment_distances,
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
    plt.savefig(f"{reward.figure_path}{title}.png", dpi=600)
    plt.close()


def plot_bradley_terry(data1, title, data2=None):
    if data2 is not None:
        sns.histplot(data1, kde=True, color="b", label="Ground Truth")
        sns.histplot(data2, kde=True, color="r", label="Trained", alpha=0.5)
        plt.legend()
    else:
        sns.histplot(data1, kde=True)
    plt.title(title)
    plt.savefig(f"{reward.figure_path}{title}.png", dpi=600)
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
    plt.savefig(f"{reward.figure_path}{title}.png", dpi=600)
    plt.close()


def handle_plotting_rei(
    model_info,
    true_agent_satisfaction_segments,
    true_agent_rewards,
    trained_agent_satisfaction_segments,
    trained_agent_rewards,
    trained_segment_rules_satisifed,
    trained_segment_rewards,
    trained_segment_distances,
    training_segment_rules_satisfied,
    training_segment_rewards,
    training_segment_distances,
):
    epochs = model_info["epochs"]
    pairs_learned = model_info["pairs-learned"]

    agents_per_generation = len(true_agent_satisfaction_segments[0])
    # Avg/max number of satisfaction segments per trajectory for each gt agent over generations
    trueRF_average_satisfaction_segments = [
        (sum(generation) / agents_per_generation)
        for generation in true_agent_satisfaction_segments
    ]
    trueRF_max_satisfaction_segments = [
        max(generation) for generation in true_agent_satisfaction_segments
    ]

    # Avg/max number of satisfaction segments per trajectory for each trained agent over generations
    trainedRF_average_satisfaction_segments = [
        (sum(generation) / agents_per_generation)
        for generation in trained_agent_satisfaction_segments
    ]
    trainedRF_max_satisfaction_segments = [
        max(generation) for generation in trained_agent_satisfaction_segments
    ]

    # Avg/max reward obtained by gt agents over generations
    true_agent_reward_averages = [
        (sum(generation) / agents_per_generation) for generation in true_agent_rewards
    ]
    true_agent_reward_maxes = [max(generation) for generation in true_agent_rewards]

    # Avg/max reward obtained by trained agents over generations
    trained_agent_reward_averages = [
        (sum(generation) / agents_per_generation)
        for generation in trained_agent_rewards
    ]
    trained_agent_reward_maxes = [
        max(generation) for generation in trained_agent_rewards
    ]

    graph_satisfaction_segments_over_generations(
        (trueRF_average_satisfaction_segments, trainedRF_average_satisfaction_segments),
        (trueRF_max_satisfaction_segments, trainedRF_max_satisfaction_segments),
    )

    graph_against_trained_reward(
        (true_agent_reward_averages, trained_agent_reward_averages),
        (true_agent_reward_maxes, trained_agent_reward_maxes),
    )

    graph_segment_rules_vs_reward(
        "retrained",
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        epochs,
        pairs_learned,
    )

    graph_segment_rules_vs_reward(
        "training",
        training_segment_rules_satisfied,
        training_segment_rewards,
    )

    graph_segment_distance_vs_reward(
        "Agent Segment Distance vs. Reward",
        trained_segment_distances,
        trained_segment_rewards,
        epochs,
        pairs_learned,
    )

    graph_segment_distance_vs_reward(
        "Training Segment Distances vs. Reward",
        training_segment_distances,
        training_segment_rewards,
    )


def graph_satisfaction_segments_over_generations(averages, maxes):
    trueRF_average_satisfaction_segments, trainedRF_average_satisfaction_segments = (
        averages
    )
    trueRF_max_satisfaction_segments, trainedRF_max_satisfaction_segments = maxes

    os.makedirs(reward.figure_path, exist_ok=True)

    x_values = range(len(trainedRF_average_satisfaction_segments))

    plt.figure()
    plt.plot(x_values, trueRF_average_satisfaction_segments, label="Ground Truth Agent")
    plt.plot(x_values, trainedRF_average_satisfaction_segments, label="Trained Agent")
    plt.xlabel("Generation")
    plt.ylabel("Number of Expert Trajectories")
    plt.title("Ground Truth vs Trained Agent: Average Ground Truth Reward")
    plt.legend()
    plt.savefig(f"{reward.figure_path}average.png", dpi=600)
    plt.close()

    plt.figure()
    plt.plot(x_values, trueRF_max_satisfaction_segments, label="Ground Truth Agent")
    plt.plot(x_values, trainedRF_max_satisfaction_segments, label="Trained Agent")
    plt.xlabel("Generation")
    plt.ylabel("Number of Expert Trajectories")
    plt.title("Ground Truth vs Trained Agents: Max Ground Truth Reward")
    plt.legend()
    plt.savefig(f"{reward.figure_path}max.png", dpi=600)
    plt.close()


def graph_against_trained_reward(averages, maxes):
    true_agent_reward_averages, trained_agent_reward_averages = averages
    true_agent_reward_maxes, trained_agent_reward_maxes = maxes

    os.makedirs(reward.figure_path, exist_ok=True)

    x_values = range(len(true_agent_reward_averages))

    plt.figure()
    plt.plot(x_values, true_agent_reward_averages, label="GT Agent")
    plt.plot(x_values, trained_agent_reward_averages, label="Trained Agent")
    plt.xlabel("Generation")
    plt.ylabel("Avg. Fitness Per Generation")
    plt.title("Avg. Fitness wrt Trained Reward")
    plt.legend()
    plt.savefig(f"{reward.figure_path}average_trained_reward.png", dpi=600)
    plt.close()

    plt.figure()
    plt.plot(x_values, true_agent_reward_maxes, label="Ground Truth Agent")
    plt.plot(x_values, trained_agent_reward_maxes, label="Trained Agent")
    plt.xlabel("Generation")
    plt.ylabel("Max Fitness Per Generation")
    plt.title("Max Fitness wrt Trained Reward")
    plt.legend()
    plt.savefig(f"{reward.figure_path}max_trained_reward.png", dpi=600)
    plt.close()


def graph_segment_rules_vs_reward(
    segment_type,
    segment_rules_satisfied,
    segment_rewards,
    epochs=None,
    pairs_learned=None,
):
    title = ""
    if segment_type == "retrained":
        title = "Retrained Segments: Rules vs Reward"
    elif segment_type == "training":
        title = "Training Segments: Rules vs Reward"

    if not segment_rules_satisfied or not segment_rewards:
        return

    if len(segment_rules_satisfied) != len(segment_rewards):
        print(
            "Graphing Segment Rules vs. Reward Error: Sizes of input lists do not match!"
        )
        return

    rewards_for_rules = [[] for _ in range(rules.NUMBER_OF_RULES + 1)]

    for i in range(len(segment_rules_satisfied)):
        num_rules, segment_reward = (
            segment_rules_satisfied[i],
            segment_rewards[i],
        )
        rewards_for_rules[num_rules].append(segment_reward)

    print(title)
    for i in range(rules.NUMBER_OF_RULES + 1):
        print("Rules Satisfied:", i, "| COUNT:", len(rewards_for_rules[i]))

    df = pd.DataFrame(
        {
            "Rules Satisfied": segment_rules_satisfied,
            "Reward of Trajectory Segment": segment_rewards,
        }
    )

    sns.violinplot(
        x="Rules Satisfied",
        y="Reward of Trajectory Segment",
        data=df,
        inner="box",
        palette="muted",
        alpha=0.55,
    )
    
    plt.title(title)
    plt.legend()
    plt.savefig(f"{reward.figure_path}{title}.png", dpi=600)
    plt.close()

    save = segment_type == "retrained"
    log_wrong(segment_rules_satisfied, segment_rewards, save)


def log_wrong(segment_rules_satisfied, segment_rewards, save=False):
    zipped_rules_reward = list(zip(segment_rules_satisfied, segment_rewards))
    random.shuffle(zipped_rules_reward)
    if len(zipped_rules_reward) % 2 != 0:
        zipped_rules_reward.pop()
    pairs_of_zips = [
        (
            zipped_rules_reward[i],
            zipped_rules_reward[i + 1],
            (zipped_rules_reward[i][0] == rules.NUMBER_OF_RULES)
            < (zipped_rules_reward[i + 1][0] == rules.NUMBER_OF_RULES),
        )
        for i in range(0, len(zipped_rules_reward), 2)
    ]

    wrong = []
    total_correct, total_different = 0, 0
    total_correct_and_different = 0
    for zip1, zip2, label in pairs_of_zips:
        correct, different = bool(zip1[1] < zip2[1]) == bool(label), zip1[0] != zip2[0]
        if not correct and different:
            wrong.append((zip1, zip2))
        if correct and different:
            total_correct_and_different += 1
        total_correct += correct
        total_different += different

    acc = total_correct / len(pairs_of_zips)
    reacc = total_correct_and_different / (total_different)

    print("ACCURACY", acc)
    print("ACCURACY W/O SAME REWARD PAIRS", reacc)

    # print("Wrong when Different Reward:")
    # count = 0
    # for zip1, zip2 in wrong:
    #     count += 1
    #     print(
    #         f"TRUE REWARDS: {zip1[0]:11.8f}, {zip2[0]:11.8f} | MODEL REWARDS: {zip1[1]:11.8f}, {zip2[1]:11.8f}"
    #     )
    #     if count > 10:
    #         break
    # print("------------------------------------------------------------------\n")

    # if save:
    #     accs = {
    #         "acc": acc,
    #         "adjusted_acc": reacc,
    #     }
    #     with open(
    #         agent.trajectories_path + "testing_acc.pkl",
    #         "wb",
    #     ) as f:
    #         pickle.dump(accs, f)


def graph_segment_distance_vs_reward(
    title, segment_distances, segment_rewards, epochs=None, pairs_learned=None
):
    if not segment_distances or not segment_rewards:
        return

    zipped_distance_reward = list(zip(segment_distances, segment_rewards))
    distRewards = defaultdict(list)
    for distance, segment_reward in zipped_distance_reward:
        rounded_distance = round(distance, 1)
        distRewards[rounded_distance].append(segment_reward)

    print(title)
    sorted_distances = sorted(list(distRewards.keys()))
    # for sorted_distance in sorted_distances:
    #     print("DISTANCE:", sorted_distance, "| COUNT:", len(distRewards[sorted_distance]))

    avg_reward_for_distances = []
    for dist in sorted_distances:
        if len(distRewards[dist]) < 2:
            del distRewards[dist]
            sorted_distances.remove(dist)

    for rounded_distance in sorted_distances:
        avg_reward_for_distances.append(statistics.mean(distRewards[rounded_distance]))

    os.makedirs(reward.figure_path, exist_ok=True)
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
    plt.savefig(f"{reward.figure_path}{title}.png", dpi=600)
    plt.close()


def plot_rules_followed_distribution(rules_followed, title):
    rule_descriptions = {
        0: "No Rules Followed",
        1: "Distance > 30",
        2: "Left Radar > Right Radar",
        3: "Actions Different",
    }

    # Flatten the list of rules
    flattened = [num for sublist in rules_followed for num in sublist]
    number_counts = Counter(flattened)

    # Create a color map for each rule
    colors = plt.cm.tab10(range(len(number_counts)))

    # Plot each rule with a unique color
    plt.bar(number_counts.keys(), number_counts.values(), color=colors)
    plt.xlabel("Rule Number")
    plt.ylabel("Frequency")
    plt.title(title)

    # Create a legend using rule descriptions
    legend_handles = [
        mpatches.Patch(color=colors[i], label=f"Rule {rule}: {rule_descriptions[rule]}")
        for i, rule in enumerate(number_counts.keys())
    ]
    plt.legend(handles=legend_handles, title="Rules")

    # Save and close the plot
    plt.savefig(f"{reward.figure_path}{title}.png", dpi=600)


def load_models(reward_paths):
    if len(reward_paths) == 1:
        print("\nLoading reward network...")
        reward_network = TrajectoryRewardNet(
            NET_SIZE,
            hidden_size=hidden_size,
        ).to(device)
        weights = torch.load(reward_paths[0], map_location=torch.device(f"{device}"))
        reward_network.load_state_dict(weights)
        return reward_network, None
    else:
        print(f"\nLoading ensemble of {len(reward_paths)} models...")
        if reward_paths[0] == "QUICK":
            if len(reward_paths) > 2:
                raise Exception("REWARD PATH ERROR (QUICK MODE)")
            reward_paths = []
            for file in glob.glob(reward_paths[1]):
                reward_paths.append(file)

        ensemble_nets = [
            TrajectoryRewardNet(
                NET_SIZE,
                hidden_size=hidden_size,
            ).to(device)
            for _ in range(len(reward_paths))
        ]
        ensemble_weights = []
        for reward_path in reward_paths:
            ensemble_weights.append(
                torch.load(reward_path, map_location=torch.device(f"{device}"))
            )
        for i in range(len(ensemble_nets)):
            ensemble_nets[i].load_state_dict(ensemble_weights[i])
            print(f"Loaded model #{i} from ensemble...")
        ensemble = Ensemble(NET_SIZE * 2, len(ensemble_nets), ensemble_nets)
        return None, ensemble


if __name__ == "__main__":
    run_wandb = False
    parse = argparse.ArgumentParser(description="Generating Plots for trained model")
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
        action="append",
        help="Directory to reward function weights",
    )
    parse.add_argument(
        "-p",
        "--parameters",
        type=str,
        help="Directory to hyperparameter yaml file",
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
                rules.NUMBER_OF_RULES = int(training_database.split("_")[3])

        except Exception:
            pass
    epochs = None
    if args.reward:
        reward = args.reward
        epochs = reward.split("_")[1].split(".")[0]

    with open(
        args.parameters if args.parameters is not None else "best_params.yaml", "r"
    ) as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]

    reward_network, ensemble = load_models(args.reward, hidden_size)
    model_info = {
        "net": reward_network,
        "ensemble": ensemble,
        "hidden-size": hidden_size,
        "epochs": epochs,
        "pairs-learned": num_pairs_learned,
        "agents-per-generation": 20,
    }

    (
        true_agent_satisfaction_segments,
        true_agent_rewards,
        trained_agent_satisfaction_segments,
        trained_agent_rewards,
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        trained_segment_distances,
        training_segment_rules_satisfied,
        training_segment_rewards,
        training_segment_distances,
    ) = populate_lists(
        true_database,
        trained_database,
        training_database,
        model_info,
    )

    handle_plotting_rei(
        model_info,
        true_agent_satisfaction_segments,
        true_agent_rewards,
        trained_agent_satisfaction_segments,
        trained_agent_rewards,
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        trained_segment_distances,
        training_segment_rules_satisfied,
        training_segment_rewards,
        training_segment_distances,
    )

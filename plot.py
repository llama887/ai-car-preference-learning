from reward import (
    TrajectoryRewardNet,
    prepare_single_trajectory,
)
import reward

figure_path = reward.figure_path

import pickle
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse

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


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Generatig Plots for trained model")
    parse.add_argument(
        "-d", "--database", type=str, help="Directory to trajectory database file"
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
    if args.reward:
        reward = args.reward

    bt, bt_, bt_delta, ordered_trajectories = prepare_data(
        database, reward, hidden_size=592
    )
    plot_bradley_terry(bt, "False Bradley Terry", bt_)
    plot_bradley_terry(bt_delta, "Bradley Terry Difference")
    plot_trajectory_order(ordered_trajectories, "Trajectory Order")

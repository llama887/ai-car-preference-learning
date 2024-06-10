from reward import (
    TrajectoryRewardNet,
    prepare_single_trajectory,
)
import pickle
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def prepare_data(database_path, model_weights):
    with open(database_path, "rb") as f:
        trajectories = pickle.load(f)
    reward1 = [t[-2] for t in trajectories]
    reward2 = [t[-1] for t in trajectories]
    rewards = reward1 + reward2
    normalizer = RewardNormalizer(rewards)
    normalized_true_reward1 = normalizer.get_normalized_rewards()[: len(reward1)]
    normalized_true_reward2 = normalizer.get_normalized_rewards()[len(reward1) :]

    fn = []
    fp = []
    false_bradley_terry = []
    bradley_terry_difference = []

    hidden_size = re.search(r"best_model_(\d+)\.pth", model_weights)
    model = TrajectoryRewardNet(900, hidden_size=int(hidden_size.group(1))).to(device)
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    for (
        index,
        t,
    ) in enumerate(trajectories):
        (
            trajectory1,
            trajectory2,
            true_preference,
            _,
            _,
        ) = t
        reward1 = normalizer.normalize(model(prepare_single_trajectory(trajectory1)))
        reward2 = normalizer.normalize(model(prepare_single_trajectory(trajectory2)))
        predicted_bradley_terry = bradley_terry(reward1, reward2)
        bradley_terry_difference.append(
            predicted_bradley_terry
            - bradley_terry(
                normalized_true_reward1[index], normalized_true_reward2[index]
            )
        )

        preference = 1 if reward1 > reward2 else 0
        if preference == true_preference:
            continue
        false_bradley_terry.append(predicted_bradley_terry)
        if preference == 1:
            fp.append(trajectory1)
            fn.append(trajectory2)
        else:
            fp.append(trajectory2)
            fn.append(trajectory1)
    return fn, fp, false_bradley_terry, bradley_terry_difference


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
    plt.savefig(f"figures/{title}.png")
    plt.close()


def plot_bradley_terry(data, title):
    sns.histplot(data, kde=True)
    plt.title(title)
    plt.savefig(f"figures/{title}.png")
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
    args = parse.parse_args()
    if args.database:
        database = args.database
    if args.reward:
        reward = args.reward

    fn, fp, bt, bt_delta = prepare_data(database, reward)
    plot_trajectories(fp, "False Positives Trajectories")
    plot_trajectories(fn, "False Negatives Trajectories")
    plot_bradley_terry(bt, "False Bradley Terry")
    plot_bradley_terry(bt_delta, "Bradley Terry Difference")

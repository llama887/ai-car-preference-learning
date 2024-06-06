from reward import (
    TrajectoryRewardNet,
    prepare_single_trajectory,
)
import pickle
import re
import torch
import matplotlib.pyplot as plt

with open("./trajectories/database_50.pkl", "rb") as f:
    trajectories = pickle.load(f)
fn = []
fp = []
model_weights = "best_model_409.pth"
hidden_size = re.search(r"best_model_(\d+)\.pth", model_weights)
model = TrajectoryRewardNet(900, hidden_size=int(hidden_size.group(1)))
model.load_state_dict(torch.load(model_weights))
model.eval()
for trajectory1, trajectory2, true_preference, _, _ in trajectories:
    reward1 = model(prepare_single_trajectory(trajectory1))
    reward2 = model(prepare_single_trajectory(trajectory2))
    preference = 1 if reward1 > reward2 else 0
    if preference == true_preference:
        continue
    if preference == 1:
        fp.append(trajectory1)
        fn.append(trajectory2)
    else:
        fp.append(trajectory2)
        fn.append(trajectory1)


# Define a function to plot trajectories
def plot_trajectories(trajectories, title):
    plt.figure(figsize=(10, 6))
    for idx, trajectory in enumerate(trajectories):
        xs, ys = zip(*trajectory)  # Unpack x and y coordinates
        plt.plot(xs, ys, alpha=0.7)  # Plot the trajectory with markers
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(f"figures/{title}.png")


# Plot false positives
plot_trajectories(fp, "False Positives Trajectories")
# Plot false negatives
plot_trajectories(fn, "False Negatives Trajectories")

import reward
from reward import TrajectoryRewardNet, prepare_single_trajectory

figure_path = reward.figure_path

import argparse

import heapq
import matplotlib.pyplot as plt
import torch
import pickle
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NET_SIZE = 4

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

def display_top_trajectories(database_path, num, best, model_weights=None, hidden_size=None):
    traj_heap = []
    if model_weights is not None:
        model = TrajectoryRewardNet(NET_SIZE, hidden_size=int(hidden_size)).to(device)
        model.load_state_dict(torch.load(model_weights))
    with open(database_path, "rb") as f:
        trajectories = pickle.load(f)
        for trajectory in trajectories:
            total_reward = 0
            total_distance = 0
            for segment in break_into_segments(trajectory[0]):
                if not model_weights:
                    total_reward += dist(segment)
                    total_distance += dist(segment)
                else:
                    total_reward += model(prepare_single_trajectory(segment)).item()
                    total_distance += dist(segment)
            if best:
                total_reward *= -1
            heapq.heappush(traj_heap, (total_reward, trajectory[0], total_distance))
            total_reward = 0
            total_distance = 0
            for segment in break_into_segments(trajectory[1]):
                if not model_weights:
                    total_reward += dist(segment)
                    total_distance += dist(segment)
                else:
                    total_reward += model(prepare_single_trajectory(segment)).item()
                    total_distance += dist(segment)
            if best:
                total_reward *= -1
            heapq.heappush(traj_heap, (total_reward, trajectory[1], total_distance))
    top_traj = []
    for i in range(num):
        if not traj_heap:
            break
        traj = heapq.heappop(traj_heap)
        top_traj.append(traj[1])
        if best:
            traj = (traj[0] * -1, traj[1], traj[2])
        print(f'{i}. FITNESS (TOTAL REWARD): {traj[0]} | DIST: {traj[2]}')
    graph_top_traj(top_traj)

def graph_top_traj(top_traj):
    for traj in top_traj:
        x, y = zip(*traj)  # Unzipping the data into x and y coordinates
        plt.plot(x, y, marker='o', markersize=2)  # Plotting the data with markers

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Top Trajs')
    plt.legend([f'Traj #{i+1}' for i in range(len(top_traj))])
    plt.savefig(f"{figure_path}/best_trajectories.png")
    plt.close()



if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Generatig Plots for trained model")
    parse.add_argument(
        "-d",
        "--database",
        type=str,
        help="Directory to trajectory database file",
    )
    parse.add_argument(
        "-r",
        "--reward",
        type=str,
        help="Directory to Directory to reward function weights",
    )
    parse.add_argument(
        "-n",
        "--number",
        type=int,
        default=5,
        help="top # trajectories",
    )
    parse.add_argument(
        "-b",
        "--best",
        action="store_true",
        help="flag for big mode",
    )
    args = parse.parse_args()
    if args.database:
        database = args.database
    if args.reward:
        reward = args.reward
    if args.number:
        num = args.number

    display_top_trajectories(
        database, num, args.best, model_weights = reward, hidden_size = 558
    )
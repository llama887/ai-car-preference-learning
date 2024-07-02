import argparse
import glob
import os
import random
import re
import sys

import matplotlib.pyplot as plt
import torch
import wandb

import agent
from agent import TRAIN_TRAJECTORY_LENGTH, run_population, trajectory_path
from plot import handle_plotting, populate_lists
from reward import TrajectoryRewardNet, train_reward_function

# from plot import prepare_data, plot_bradley_terry, plot_trajectory_order

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_SILENT"] = "true"


def start_simulation(
    config_path, max_generations, number_of_trajectories, run_type, noHead
):
    # Set number of trajectories
    agent.number_of_trajectories = number_of_trajectories

    return run_population(
        config_path=config_path,
        max_generations=max_generations,
        number_of_trajectories=number_of_trajectories,
        runType=run_type,
        noHead=noHead,
    )


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )
    parse.add_argument(
        "-e",
        "--epochs",
        type=int,
        nargs=1,
        help="Number of epochs to train the model",
    )
    parse.add_argument(
        "-t",
        "--trajectories",
        type=int,
        nargs=1,
        help="Number of trajectories to collect",
    )
    parse.add_argument(
        "-g",
        "--generations",
        type=int,
        nargs=1,
        help="Number of generations to train the agent",
    )
    parse.add_argument(
        "-p",
        "--parameters",
        type=str,
        help="Directory to hyperparameter yaml file",
    )
    parse.add_argument(
        "--headless", action="store_true", help="Run simulation without GUI"
    )

    args = parse.parse_args()
    if (
        args.trajectories[0] < 0
        or args.generations[0] < 0
        or args.epochs[0] < 0
    ):
        print("Invalid input. All arguments must be positive integers.")
        sys.exit(1)
    # if args.headless:
    #     os.environ["SDL_VIDEODRIVER"] = "dummy"
    # start the simulation in data collecting mode
    num_traj = start_simulation(
        "./config/data_collection_config.txt",
        args.trajectories[0],
        args.trajectories[0],
        "collect",
        args.headless,
    )

    database_path = f"trajectories/database_{num_traj}.pkl"

    print("Starting training on trajectories...")
    train_reward_function(database_path, args.epochs[0], args.parameters)
    print("Finished training model...")

    # run the simulation with the true reward function
    print("Simulating on true reward function...")
    truePairs = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trueRF",
        False,
    )

    # run the simulation with the trained reward function
    print("Simulating on trained reward function...")
    agent.reward_network = TrajectoryRewardNet(TRAIN_TRAJECTORY_LENGTH * 2).to(
        device
    )
    trainedPairs = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trainedRF",
        False,
    )

    # true_agent_distances, agent_trajectories, true_agent_rewards = (
    #     [[0] * len(trained_agent_distances[0])] * len(trained_agent_distances),
    #     [0],
    #     [[0] * len(trained_agent_rewards[0])] * len(trained_agent_rewards),
    # )

    true_database = trajectory_path + f"trueRF_{truePairs}.pkl"
    trained_database = trajectory_path + f"trainedRF_{trainedPairs}.pkl"
    model_weights = f"model_{args.epochs[0]}.pth"
    (
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
    ) = populate_lists(
        true_database,
        trained_database,
        agents_per_generation=20,
        model_weights=model_weights,
        hidden_size=311,
    )

    print("PLOTTING...")
    handle_plotting(
        true_agent_distances,
        trained_agent_distances,
        trained_agent_rewards,
        trained_segment_distances,
        trained_segment_rewards,
    )

    # bt, bt_delta, ordered_trajectories = prepare_data(
    #     f"trajectories/trainedRF_{agent_trajectories}.pkl", net=agent.reward_network
    # )
    # plot_bradley_terry(bt, "Agent False Bradley Terry")
    # plot_bradley_terry(bt_delta, "Agent Bradley Terry Difference")
    # plot_trajectory_order(ordered_trajectories, "Trajectory Order")

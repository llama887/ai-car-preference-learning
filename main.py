import argparse

from reward import TrajectoryRewardNet, run_study

import agent
from agent import run_population, TRAJECTORY_LENGTH

import matplotlib.pyplot as plt
import wandb
import glob
import os
import sys
import re


os.environ["WANDB_SILENT"] = "true"


def start_simulation(config_path, max_generations, number_of_trajectories=-1):
    # Set number of trajectories
    agent.number_of_trajectories = number_of_trajectories

    return run_population(
        config_path=config_path,
        max_generations=max_generations,
        number_of_trajectories=number_of_trajectories,
    )


def handle_plotting(true_agent_distances, trained_agent_distances):
    traj_per_generation = len(true_agent_distances[0])
    num_generations = len(true_agent_distances)
    true_reward_averages = [
        (sum(generation) / traj_per_generation) for generation in true_agent_distances
    ]
    true_reward_maxes = [max(generation) for generation in true_agent_distances]

    trained_reward_averages = [
        (sum(generation) / traj_per_generation)
        for generation in trained_agent_distances
    ]
    trained_reward_maxes = [max(generation) for generation in trained_agent_distances]
    graph_avg_max(
        (true_reward_averages, trained_reward_averages),
        (true_reward_maxes, trained_reward_maxes),
    )


def graph_avg_max(averages, maxes):
    true_reward_averages, trained_reward_averages = averages
    true_reward_maxes, trained_reward_maxes = maxes

    os.makedirs("figures", exist_ok=True)

    x_values = range(len(trained_reward_averages))

    plt.figure()
    plt.plot(x_values, true_reward_averages, label="Ground Truth")
    plt.plot(x_values, trained_reward_averages, label="Trained Reward", marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("Ground Truth vs Trained Reward: Average Distance")
    plt.legend()
    plt.savefig("figures/average.png")

    plt.figure()
    plt.plot(x_values, true_reward_maxes, label="Ground Truth")
    plt.plot(x_values, trained_reward_maxes, label="Trained Reward", marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("Ground Truth vs Trained Reward: Max Distance")
    plt.legend()
    plt.savefig("figures/max.png")

    wandb.log({"Avg Plot": wandb.Image("figures/average.png")})
    wandb.log({"Max Plot": wandb.Image("figures/max.png")})

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


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )
    parse.add_argument(
        "-e", "--epochs", type=int, nargs=1, help="Number of epochs to train the model"
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
    args = parse.parse_args()
    if args.trajectories[0] < 0 or args.generations[0] < 0 or args.epochs[0] < 0:
        print("Invalid input. All arguments must be positive integers.")
        sys.exit(1)
    database_path = f"trajectories/database_{args.trajectories[0]//2}.pkl"

    print("Removing old trajectories...")
    old_trajectories = glob.glob("trajectories/trajectory*")
    for f in old_trajectories:
        os.remove(f)
    print(f"Saving {args.trajectories[0]} trajectories...")

    # # start the simulation in data collecting mode
    # start_simulation(
    #     "./config/data_collection_config.txt",
    #     args.trajectories[0],
    #     args.trajectories[0],
    # )
    start_simulation(
        "./config/data_collection_config.txt",
        args.trajectories[0],
        args.trajectories[0],
    )

    print("Starting training on trajectories...")
    # train model on collected data
    # train_model(database_path, epochs=args.epochs)
    run_study(database_path, args.epochs[0])
    print("Finished training model...")

    print("Simulating on true reward function...")
    # run the simulation with the true reward function
    # true_reward_averages, true_reward_maxes = start_simulation(
    #     "./config/agent_config.txt",
    #     args.generations[0],
    # )
    true_agent_distances = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
    )
    print("Simulating on trained reward function...")
    # run the simulation with the trained reward function
    agent.reward_network = TrajectoryRewardNet(TRAJECTORY_LENGTH * 2)
    # trained_reward_averages, trained_reward_maxes = start_simulation(
    #     "./config/agent_config.txt",
    #     args.generations[0],
    # )
    trained_agent_distances = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
    )
    handle_plotting(true_agent_distances, trained_agent_distances)

import argparse

from reward import TrajectoryRewardNet, train_reward_function

import agent
from agent import run_population, TRAJECTORY_LENGTH

import matplotlib.pyplot as plt
import wandb
import glob
import os
import sys
import re


os.environ["WANDB_SILENT"] = "true"


def start_simulation(
    config_path, max_generations, number_of_trajectories, run_type, noHead=True
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


def handle_plotting(true_agent_distances, trained_agent_distances):
    traj_per_generation = len(true_agent_distances[0])
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
    graph_death_rates(true_agent_distances, "GT")
    graph_death_rates(trained_agent_distances, "Trained")


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

    plt.figure()
    plt.plot(x_values, true_reward_maxes, label="Ground Truth")
    plt.plot(x_values, trained_reward_maxes, label="Trained Reward")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.title("Ground Truth vs Trained Reward: Max Distance")
    plt.legend()
    plt.savefig("figures/max.png")

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
    parse.add_argument(
        "-p", "--parameters", type=str, help="Directory to hyperparameter yaml file"
    )
    parse.add_argument(
        "--headless", action="store_true", help="Run simulation without GUI"
    )

    args = parse.parse_args()
    if args.trajectories[0] < 0 or args.generations[0] < 0 or args.epochs[0] < 0:
        print("Invalid input. All arguments must be positive integers.")
        sys.exit(1)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    # start the simulation in data collecting mode
    start_simulation(
        "./config/data_collection_config.txt",
        args.trajectories[0],
        args.trajectories[0],
        "collect",
        args.headless,
    )

    database_path = f"trajectories/database_{args.trajectories[0]//2}.pkl"

    print("Starting training on trajectories...")
    train_reward_function(database_path, args.epochs[0], args.parameters)
    print("Finished training model...")

    print("Simulating on true reward function...")
    # run the simulation with the true reward function
    true_agent_distances = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trueRF",
        False,
    )
    print("Simulating on trained reward function...")
    # run the simulation with the trained reward function
    agent.reward_network = TrajectoryRewardNet(TRAJECTORY_LENGTH * 2)
    trained_agent_distances = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trainedRF",
        False,
    )
    handle_plotting(true_agent_distances, trained_agent_distances)

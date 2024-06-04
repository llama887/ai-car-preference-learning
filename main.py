import argparse

from reward import train_model, TrajectoryRewardNet

import agent
from agent import run_simulation, TRAJECTORY_LENGTH
import neat
import glob
import os
import sys
import re


def start_simulation(config_path, max_generations, number_of_trajectories=-1):
    # Set number of trajectories
    agent.number_of_trajectories = number_of_trajectories

    # Load Config
    config_path = config_path
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(run_simulation, max_generations)


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

    # start the simulation in data collecting mode
    start_simulation(
        "./config/data_collection_config.txt",
        args.trajectories[0],
        args.trajectories[0],
    )

    # train model on collected data
    train_model(database_path, epochs=args.epochs)

    # run the simulation with the true reward function
    start_simulation("./config/agent_config.txt", args.generations[0])

    # run the simulation with the trained reward function
    agent.reward_network = TrajectoryRewardNet(TRAJECTORY_LENGTH * 2)
    start_simulation("./config/agent_config.txt", args.generations[0])

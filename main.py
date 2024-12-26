import argparse
import glob
import os
import pickle
import random
import sys
from multiprocessing import Process

import torch
import yaml

import agent
import rules
from agent import STATE_ACTION_SIZE, run_population, trajectory_path
from plot import (
    handle_plotting,
    plot_rules_followed_distribution,
    populate_lists,
)
from reward import TrajectoryRewardNet, train_reward_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_SILENT"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def start_simulation(config_path, max_generations, number_of_pairs, run_type, noHead):
    # Set number of trajectories
    agent.number_of_pairs = number_of_pairs

    return run_population(
        config_path=config_path,
        max_generations=max_generations,
        number_of_pairs=number_of_pairs,
        runType=run_type,
        noHead=noHead,
    ), agent.rules_followed


def sample_from_database(num_pairs, database_path):
    with open(database_path, "rb") as f:
        database = pickle.load(f)
    total_pairs = len(database)

    if num_pairs == total_pairs:
        return database_path
    elif num_pairs < total_pairs:
        new_pairs = random.sample(database, num_pairs)
        new_database_path = trajectory_path + f"database_{num_pairs}.pkl"
        with open(new_database_path, "wb") as f:
            pickle.dump(new_pairs, f)
        return new_database_path
    else:
        return -1


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
        help="Number of pairs of segments to collect",
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
    parse.add_argument(
        "-r",
        "--reward",
        type=str,
        help="Directory to reward function weights",
    )
    parse.add_argument(
        "-d",
        "--database",
        type=str,
        help="Directory to trajectory database file, the number of pairs indicated by this file name will override -t flag",
    )
    parse.add_argument(
        "--rules_distribution",
        type=int,
        nargs="+",
        help="Distribution of rules followed by the agent. Number of rules in the reward function is defined as len(rules_distribution)-1",
    )

    args = parse.parse_args()
    if (
        (args.trajectories is not None and args.trajectories[0] < 0)
        or (args.generations is not None and args.generations[0] < 0)
        or (args.epochs is not None and args.epochs[0] < 0)
    ):
        print("Invalid input. All arguments must be positive integers.")
        sys.exit(1)

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    database_path = ""
    if args.trajectories and args.database:
        database_path = sample_from_database(args.trajectories[0], args.database)
        if database_path == -1:
            print("Provide a larger database, or generate a new one!")
            sys.exit(1)
        num_pairs = database_path.split("_")[1].split(".")[0]
        args.trajectories[0] = num_pairs
    elif args.trajectories:
        database_path = f"trajectories/database_{args.trajectories[0]}.pkl"
    elif args.database:
        database_path = args.database
        num_pairs = database_path.split("_")[1].split(".")[0]
        args.trajectories[0] = num_pairs
    else:
        print(
            "Need to either provide number of trajectories to collect or existing database"
        )

    model_weights = ""
    if args.reward is None:
        # start the simulation in data collecting mode
        if not args.database:
            num_traj, collecting_rules_followed = start_simulation(
                "./config/data_collection_config.txt",
                args.trajectories[0],
                args.trajectories[0],
                "collect",
                args.headless,
            )
            plot_rules_followed_distribution(
                collecting_rules_followed, "Input Distribution Rules Followed"
            )

        print("Starting training on trajectories...")
        train_reward_function(database_path, args.epochs[0], args.parameters)
        print("Finished training model...")

        # run the simulation with the trained reward function

        try:
            optimized_weights = [f for f in glob.glob("best_model_*.pth")][0]
        except IndexError:
            optimized_weights = None
        model_weights = (
            f"model_{args.epochs[0]}.pth" if args.parameters else optimized_weights
        )
    else:
        model_weights = args.reward
    if args.rules_distribution:
        rules.SEGMENT_DISTRIBUTION_BY_RULES = args.rules_distribution
        rules.NUMBER_OF_RULES = len(rules.SEGMENT_DISTRIBUTION_BY_RULES) - 1
    # run the simulation with the true reward function
    print("Simulating on true reward function...")
    truePairs, true_rules_followed = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trueRF",
        args.headless,
    )

    with open(
        args.parameters if args.parameters is not None else "best_params.yaml", "r"
    ) as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]

    print("Simulating on trained reward function...")
    agent.reward_network = TrajectoryRewardNet(
        STATE_ACTION_SIZE * 2, hidden_size=hidden_size
    ).to(device)

    weights = torch.load(model_weights, map_location=device)
    agent.reward_network.load_state_dict(weights)
    trainedPairs, trained_rules_followed = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trainedRF",
        args.headless,
    )

    def plot_collecting_rules():
        if collecting_rules_followed:
            plot_rules_followed_distribution(
                collecting_rules_followed, "Input Distribution Rules Followed"
            )

    def plot_true_rules():
        plot_rules_followed_distribution(
            true_rules_followed, "Ground Truth Rules Followed"
        )

    def plot_true_expert_rules():
        plot_rules_followed_distribution(
            true_rules_followed[-10000:], "Expert Ground Truth Rules Followed"
        )

    def plot_trained_rules():
        plot_rules_followed_distribution(
            trained_rules_followed, "Trained Agent Rules Followed"
        )

    def plot_trained_expert_rules():
        plot_rules_followed_distribution(
            trained_rules_followed[-10000:], "Expert Trained Agent Rules Followed"
        )

    print("Plotting Rules Followed Distributions...")
    process_collecting = Process(target=plot_collecting_rules)
    process_true = Process(target=plot_true_rules)
    process_true_expert = Process(target=plot_true_expert_rules)
    process_trained = Process(target=plot_trained_rules)
    process_trained_expert = Process(target=plot_trained_expert_rules)
    (
        process_collecting.start(),
        process_true.start(),
        process_true_expert.start(),
        process_trained.start(),
        process_trained_expert.start(),
    )
    (
        process_collecting.join(),
        process_true.join(),
        process_true_expert.join(),
        process_trained.join(),
        process_trained_expert.join(),
    )

    model_info = {
        "weights": model_weights,
        "net": None,
        "hidden-size": hidden_size,
        "epochs": -1 if args.epochs is None else args.epochs[0],
        "pairs-learned": -1 if args.trajectories is None else args.trajectories[0],
        "agents-per-generation": 20,
    }

    true_database = trajectory_path + f"trueRF_{truePairs}.pkl"
    trained_database = trajectory_path + f"trainedRF_{trainedPairs}.pkl"
    (
        true_agent_expert_segments,
        trained_agent_expert_segments,
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
        database_path,
        model_info,
    )

    print("PLOTTING...")
    handle_plotting(
        model_info,
        true_agent_expert_segments,
        trained_agent_expert_segments,
        trained_agent_rewards,
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        trained_segment_distances,
        training_segment_rules_satisfied,
        training_segment_rewards,
        training_segment_distances,
    )

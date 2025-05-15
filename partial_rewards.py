import argparse
import os
import sys
import numpy as np
import pickle
import yaml
import torch
from torch.utils.data import DataLoader

import rules
import agent
from test_accuracy import test_model
from agent import run_population, STATE_ACTION_SIZE
from reward import (
    train_reward_function,
    TrajectoryRewardNet,
    TrajectoryDataset,
    prepare_single_trajectory
)
import reward

rules.PARTIAL_REWARD = True
rules.NUMBER_OF_RULES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_SILENT"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def start_simulation(config_path, max_generations, number_of_pairs, run_type, noHead, use_ensemble):
    # Set number of trajectories
    agent.number_of_pairs = number_of_pairs

    return (
        run_population(
            config_path=config_path,
            max_generations=max_generations,
            number_of_pairs=number_of_pairs,
            runType=run_type,
            noHead=noHead,
            use_ensemble=use_ensemble
        ),
        agent.rules_followed,
    )                


def process_distribution(args):
    a, b, c, i, num_pairs, config_path, epochs, parameters, headless, total_iterations, hidden_size, batch_size = args
    if c >= 0:
        print(f"Distribution {i + 1}/{total_iterations}")
        distributions = [[], [], []]
        distributions[0].append(a)
        distributions[1].append(b)
        distributions[2].append(c)
        rules.SEGMENT_DISTRIBUTION_BY_RULES = [a, b, c]

        agent.trajectories_path = f"trajectories_partial/trajectories_partial_{i + 1}/"
        database_path = f"{agent.trajectories_path}database_{num_pairs}_pairs_2_rules_{agent.train_trajectory_length}_length.pkl"

        # Start the simulation
        start_simulation(
            config_path, num_pairs, num_pairs, "collect", headless, False
        )

        try:
            accs = train_reward_function(
                database_path, epochs, parameters, False, None, "acc"
            )
            model_id = "".join([str(rule) for rule in rules.RULES_INCLUDED])
            test_acc, adjusted_test_acc, _ = test_model(f"models_partial_{i}/model_{model_id}_{epochs}_epochs_{num_pairs}_pairs_{rules.NUMBER_OF_RULES}_rules.pth", hidden_size, batch_size)
            accs["final_test_acc"] = test_acc
            accs["final_adjusted_test_acc"] = adjusted_test_acc
            return (a, b, c, accs)
        except Exception as e:
            print(f"Error in processing distribution {i + 1}: {e}")
    return None


def calculate_iterations(res):
    i = 0
    step = 1 / res
    for a in np.arange(0, 1 + step, step):
        for b in np.arange(0, 1 - a + step, step):
            i += 1
    return i


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
        "-r",
        "--resolution",
        type=int,
        help="Resolution of gridding",
    )
    parse.add_argument(
        "-a",
        type=float,
        help="dist for 0 rule",
    )
    parse.add_argument(
        "-b",
        type=float,
        help="dist for 1 rule",
    )
    parse.add_argument(
        "-c",
        type=float,
        help="dist for 2 rule",
    )
    parse.add_argument(
        "-i",
        type=int,
        help="index",
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
        (args.trajectories is not None and args.trajectories[0] < 0)
        or (args.epochs is not None and args.epochs[0] < 0)
    ):
        print("Invalid input. All arguments must be positive integers.")
        sys.exit(1)

    # Display simulator or not
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Check flags
    if not (args.trajectories and args.parameters):
        print("Missing either -p flag or -t flag")
        sys.exit()

    num_pairs = args.trajectories[0]
    with open(args.parameters, "r") as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]
        batch_size = data["batch_size"]

    a, b, c, i = args.a, args.b, args.c, args.i 
    reward.models_path = f"models_partial_{i}/"
    
    
    total_iterations = calculate_iterations(args.resolution)
    task =  (a, b, c, i, num_pairs, "./config/data_collection_config.txt", args.epochs[0], args.parameters, args.headless, total_iterations, hidden_size, batch_size)

    result = process_distribution(task)
    a, b, c, accs = result

    with open(
            agent.trajectories_path
            + f"result.pkl",
            "wb",
        ) as f:
            pickle.dump((a,b,c,accs), f)


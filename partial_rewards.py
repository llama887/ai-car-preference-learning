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
from agent import run_population, STATE_ACTION_SIZE
from reward import (
    train_reward_function,
    TrajectoryRewardNet,
    TrajectoryDataset,
    prepare_single_trajectory
)

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


def test_model(model_path, test_file, hidden_size, batch_size=256):
    if not model_path:
        raise Exception("Model not found...")
    if not test_file:
        raise Exception("Test file not found...")
    
    model = TrajectoryRewardNet(
        STATE_ACTION_SIZE * (agent.train_trajectory_length + 1),
        hidden_size=hidden_size,
    ).to(device)
    weights = torch.load(model_path, map_location=torch.device(f"{device}"))
    model.load_state_dict(weights)

    total_correct = 0
    total_diff = 0
    adjusted_correct = 0

    test_dataset = TrajectoryDataset(test_file, None, True)
    test_size = len(test_dataset)
    print("TEST SIZE:", test_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = test_size if test_size < batch_size else batch_size,
        shuffle=False,
        pin_memory=False,
    )

    with torch.no_grad():
        for test_traj1, test_traj2, test_true_pref, test_score1, test_score2 in test_dataloader:
            test_rewards1 = model(test_traj1)
            test_rewards2 = model(test_traj2)
            predictions = (test_rewards1 >= test_rewards2).squeeze()
            correct_predictions = (predictions == test_true_pref).sum().item()
            total_correct += correct_predictions

            different_rewards = (test_score1 != test_score2)
            num_diff_in_batch = different_rewards.sum().item()
            total_diff += num_diff_in_batch

            adjusted_correct += (different_rewards & (predictions == test_true_pref)).sum().item()
            
        test_acc = total_correct / test_size
        adjusted_test_acc = adjusted_correct / total_diff if total_diff > 0 else 0
    return test_acc, adjusted_test_acc
                



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
            test_acc, adjusted_test_acc = test_model(f"models/model_{epochs}_epochs_{num_pairs}_pairs_{rules.NUMBER_OF_RULES}_rules.pth", "database_test.pkl", hidden_size, batch_size)
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
    
    
    total_iterations = calculate_iterations(args.resolution)
    task =  (a, b, c, i, num_pairs, "./config/data_collection_config.txt", args.epochs[0], args.parameters, args.headless, total_iterations, hidden_size, batch_size)
    # test_model(f"models/model_{50}_epochs_{num_pairs}_pairs_{rules.NUMBER_OF_RULES}_rules.pth", "database_test.pkl", hidden_size, batch_size)
            
    result = process_distribution(task)
    a, b, c, accs = result

    with open(
            agent.trajectories_path
            + f"result.pkl",
            "wb",
        ) as f:
            pickle.dump((a,b,c,accs), f)


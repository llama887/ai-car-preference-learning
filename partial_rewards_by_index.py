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
from test_accuracy import test_model_light
from agent import run_population, STATE_ACTION_SIZE
from reward import (
    train_reward_function,
    TrajectoryRewardNet,
    TrajectoryDataset,
    prepare_single_trajectory
)
import reward

rules.NUMBER_OF_RULES = 2
rules.RULES_INCLUDED = [1, 2]

DISTRIBUTION_PRECISION = 8

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
        rules.PARTIAL_REWARD = True
        # Start the simulation
        start_simulation(
            config_path, num_pairs, num_pairs, "collect", headless, False
        )
        
        try:
            accs = train_reward_function(
                database_path, epochs, parameters, False, None, str(i+1), "acc"
            )
            model_id = i + 1
            test_acc, adjusted_test_acc = test_model_light([f"models_partial/model_{model_id}_{epochs}_epochs_{num_pairs}_pairs_{rules.NUMBER_OF_RULES}_rules.pth"], hidden_size, batch_size)
            accs["final_test_acc"] = test_acc
            accs["final_adjusted_test_acc"] = adjusted_test_acc
        except Exception as e:
            print(f"Error during training or testing: {e}")
        return (a, b, c, accs)
    return None

def calculate_iterations(res):
    i = 0
    step = 1 / res
    for a in np.arange(0, 1 + step, step):
        for b in np.arange(0, 1 - a + step - 1e-8, step):
            i += 1
    return i

def get_distribution(index, resolution):
    step = 1 / resolution
    i = 0
    for a in np.arange(0, 1 + step/2, step):  # Added step/2 to handle floating point precision
        for b in np.arange(0, 1 - a + step/2, step):  # Added step/2 to handle floating point precision
            if i == index:
                c = 1 - a - b
                # Round to handle floating point precision issues
                c = 0 if abs(c) < 1e-10 else c
                return a, b, c
            i += 1
    raise ValueError("Index out of range for the given resolution.")

if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )

    res = 3
    epochs = 100
    num_pairs = 1000
    param_file = "./best_params.yaml"

    
    parse.add_argument(
        "-i",
        type=int,
        help="index",
    )
    
    args = parse.parse_args()
    a, b, c = get_distribution(args.i, res)

    with open(param_file, "r") as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]
        batch_size = data["batch_size"]

    a, b, c, i = round(a, DISTRIBUTION_PRECISION), round(b, DISTRIBUTION_PRECISION), round(c, DISTRIBUTION_PRECISION), args.i 
    

    reward.models_path = f"models_partial/"
    os.makedirs(reward.models_path, exist_ok=True)
    
    total_iterations = calculate_iterations(res)
    task =  (a, b, c, i, num_pairs, "./config/data_collection_config.txt", epochs, param_file, True, total_iterations, hidden_size, batch_size)

    result = process_distribution(task)
    a, b, c, accs = result

    with open(
            agent.trajectories_path
            + f"result.pkl",
            "wb",
        ) as f:
            pickle.dump((a,b,c,accs), f)


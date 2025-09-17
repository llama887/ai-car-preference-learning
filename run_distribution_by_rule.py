import argparse
import os
import pickle
import sys

import torch
from torch.utils.data import DataLoader
import yaml
import matplotlib.pyplot as plt

import agent
import reward
import rules
from agent import (
    run_population,
    STATE_ACTION_SIZE
)
from test_accuracy import test_model_light
from reward import (
    train_reward_function,
    TrajectoryDataset,
    TrajectoryRewardNet
)

TRAJECTORY_PAIRS=1000000
EPOCHS=100000
GENERATIONS=200
RULES_LIST = [1,2,3]

param_file = "./best_params.yaml"
use_ensemble = False
headless = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_SILENT"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"

def generate_distribution(rules, satis):
    distributions = {}
    satisfaction = [0.1, 0.2, 0.3]
    non_satisfaction = [(1 - s) / rules for s in satisfaction]
    # for i in range(resolution):
    #     satisfaction = i / (resolution - 1)
    #     non_satisfaction = (resolution - 1 - i) / ((resolution - 1) * rules)
    #     distribution = []
    #     for i in range(rules):
    #         distribution.append(non_satisfaction)
    #     distribution.append(satisfaction)
    #     distributions[satisfaction] = distribution
    for s, ns in zip(satisfaction, non_satisfaction):
        distribution = [ns] * rules + [s]
        distributions[s] = distribution
    return distributions

def plot_data(figure_folder, data_points):
    plt.figure()
    for num_rules in range(1, 4):
        plt.plot(data_points[num_rules][0], data_points[num_rules][1],label=f"{num_rules} rules")
    plt.xlabel("% Satisfaction Segments")
    plt.ylabel("Adjusted Testing Accuracy")
    plt.legend()
    plt.savefig(f"{figure_folder}distribution.png", dpi=600)
    plt.close()

def start_simulation(
    config_path,
    max_generations,
    number_of_pairs,
    run_type,
    noHead=True,
    use_ensemble=False,
):
    # Set number of trajectories
    agent.number_of_pairs = number_of_pairs

    return (
        run_population(
            config_path=config_path,
            max_generations=max_generations,
            number_of_pairs=number_of_pairs,
            runType=run_type,
            noHead=noHead,
            use_ensemble=use_ensemble,
        ),
        agent.rules_followed,
    )

if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )

    parse.add_argument(
        "-r",
        "--resolution",
        type=int,
    )

    parse.add_argument(
        "-c",
        "--composition",
        type=int,
    )

    parse.add_argument(
        "-s",
        "--satisfaction",
        type=float,
    )
    

    args = parse.parse_args()

    reward.models_path = "models_dist_exp/"
    os.makedirs(reward.models_path, exist_ok=True)

    hidden_size = ""
    batch_size = ""
    with open(
            param_file, "r"
        ) as file:
            data = yaml.safe_load(file)
            hidden_size = data["hidden_size"]
            batch_size = data["batch_size"]

    data_points = ""
    num_rules = args.composition
    rules.NUMBER_OF_RULES = num_rules
    rules.RULES_INCLUDED = [i for i in range(1, rules.NUMBER_OF_RULES + 1)]
    distributions = generate_distribution(num_rules, args.satisfaction)
    total_distributions = len(distributions)
    data_x = sorted(distributions.keys())
    data_y = []
    for i, satis in enumerate(data_x):
        rules.SEGMENT_DISTRIBUTION_BY_RULES = distributions[satis]
        print(f"CURRENT DISTRIBUTION ({i + 1} / {total_distributions}):", distributions[satis])
        database_path = f"{agent.trajectories_path}database_{TRAJECTORY_PAIRS}_pairs_{rules.NUMBER_OF_RULES}_rules_{agent.train_trajectory_length}_length.pkl"
        model_weights = ""

        # start the simulation in data collecting mode
        num_traj, collecting_rules_followed = start_simulation(
            "./config/data_collection_config.txt",
            TRAJECTORY_PAIRS,
            TRAJECTORY_PAIRS,
            "collect",
            headless,
            use_ensemble,
        )

        model_id = str(satis)
        print("Starting training on trajectories...")
        final_val_acc = train_reward_function(
            trajectories_file_path=database_path,
            epochs=EPOCHS,
            parameters_path=param_file,
            use_ensemble=use_ensemble,
            figure_folder_name=None,
            model_id=model_id,
            return_stat="acc",
        )["final_adjusted_validation_acc"]
        print("Finished training model...")

        model_path = [(reward.models_path + f"model_{model_id}_{EPOCHS}_epochs_{TRAJECTORY_PAIRS}_pairs_{rules.NUMBER_OF_RULES}_rules.pth")]
        test_acc, adjusted_test_acc = test_model_light(
            model_path, hidden_size, batch_size
        )
        data_y.append(final_val_acc)
    data_points = (data_x.copy(), data_y.copy())

    figure_folder = "distribution_experiment/"
    os.makedirs(figure_folder, exist_ok=True)
    with open(
            figure_folder + f"output_{num_rules}.pkl",
            "wb",
        ) as f:
            pickle.dump(data_points, f)
    
    # plot_data(figure_folder, data_points)



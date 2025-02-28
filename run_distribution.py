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
from reward import (
    train_reward_function,
    TrajectoryDataset,
    TrajectoryRewardNet
)


TRAJECTORIES=1000000
EPOCHS=3000
GENERATIONS=200
RULES_LIST = [1,2,3]

param_file = "./best_params.yaml"
use_ensemble = False
headless = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["WANDB_SILENT"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"

def generate_distribution(resolution, rules):
    distributions = {}
    for i in range(resolution):
        satisfaction = i / (resolution - 1)
        non_satisfaction = (resolution - 1 - i) / ((resolution - 1) * rules)
        distribution = []
        for i in range(rules):
            distribution.append(non_satisfaction)
        distribution.append(satisfaction)
        distributions[satisfaction] = distribution
    return distributions

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

    args = parse.parse_args()

    reward.models_path = "models_dist_exp/"

    hidden_size = ""
    batch_size = ""
    with open(
            param_file, "r"
        ) as file:
            data = yaml.safe_load(file)
            hidden_size = data["hidden_size"]
            batch_size = data["batch_size"]

    data_points = {}
    for num_rules in RULES_LIST:
        rules.NUMBER_OF_RULES = num_rules
        distributions = generate_distribution(args.resolution, num_rules)
        total_distributions = len(distributions) * len(RULES_LIST)
        data_x = sorted(distributions.keys())
        data_y = []
        for i, satis in enumerate(data_x):
            rules.SEGMENT_DISTRIBUTION_BY_RULES = distributions[satis]
            print(f"CURRENT DISTRIBUTION ({len(distributions) * (num_rules - 1) + i + 1} / {total_distributions}):", distributions[satis])
            database_path = f"{agent.trajectories_path}database_{TRAJECTORIES}_pairs_{rules.NUMBER_OF_RULES}_rules_{agent.train_trajectory_length}_length.pkl"
            model_weights = ""

            # start the simulation in data collecting mode
            num_traj, collecting_rules_followed = start_simulation(
                "./config/data_collection_config.txt",
                TRAJECTORIES,
                TRAJECTORIES,
                "collect",
                headless,
                use_ensemble,
            )

            print("Starting training on trajectories...")
            train_reward_function(
                trajectories_file_path=database_path,
                epochs=EPOCHS,
                parameters_path=param_file,
                use_ensemble=use_ensemble,
                figure_folder_name=None,
                return_stat=None,
            )
            print("Finished training model...")

            model_path = (reward.models_path + f"model_{EPOCHS}_epochs_{TRAJECTORIES}_pairs_{rules.NUMBER_OF_RULES}_rules.pth")
            test_path = f"database_test_{rules.NUMBER_OF_RULES}_rules.pkl"
            test_acc, adjusted_test_acc = test_model(model_path, test_path, hidden_size, batch_size)
            data_y.append(adjusted_test_acc)
        data_points[num_rules] = (data_x.copy(), data_y.copy())

        figure_folder = "distribution_experiment/"
        os.makedirs(figure_folder, exist_ok=True)
        with open(
                figure_folder + "output.pkl",
                "wb",
            ) as f:
                pickle.dump(data_points, f)
        
    figure_folder = "distribution_experiment/"
    os.makedirs(figure_folder, exist_ok=True)
    with open(
            figure_folder + "output.pkl",
            "wb",
        ) as f:
            pickle.dump(data_points, f)

    plot_data(figure_folder, data_points)



import argparse
import glob
import os
import sys

import torch
import yaml

import agent
from agent import STATE_ACTION_SIZE, run_population, trajectory_path
from plot import (
    handle_plotting,
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

    args = parse.parse_args()
    if args.trajectories[0] < 0 or args.generations[0] < 0 or args.epochs[0] < 0:
        print("Invalid input. All arguments must be positive integers.")
        sys.exit(1)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    database_path = ""
    if args.database is None:
        database_path = f"trajectories/database_{args.trajectories[0]}.pkl"
    else:
        database_path = args.database
        num_pairs = database_path.split("_")[1].split(".")[0]
        args.trajectories[0] = num_pairs

    model_weights = ""
    if args.reward is None:
        # start the simulation in data collecting mode
        if not args.database:
            num_traj = start_simulation(
                "./config/data_collection_config.txt",
                args.trajectories[0],
                args.trajectories[0],
                "collect",
                args.headless,
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

    # run the simulation with the true reward function
    print("Simulating on true reward function...")
    truePairs = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trueRF",
        False,
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
    trainedPairs = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trainedRF",
        False,
    )

    model_info = {
        "weights": model_weights,
        "net": None,
        "hidden-size": hidden_size,
        "epochs": args.epochs[0],
        "pairs-learned": args.trajectories[0],
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

    # bt, bt_, bt_delta, ordered_trajectories = prepare_data(
    #     f"trajectories/trainedRF_{trainedPairs}.pkl", net=agent.reward_network
    # )

    # plot_bradley_terry(bt, "False Bradley Terry", bt_)
    # plot_bradley_terry(bt_delta, "Bradley Terry Difference")
    # plot_trajectory_order(ordered_trajectories, "Segment Order")

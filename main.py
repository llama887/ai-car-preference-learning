import argparse
import glob
import os
import pickle
import random
import sys

import torch
import yaml

import agent
import rules

from rules import NUMBER_OF_RULES, SEGMENT_DISTRIBUTION_BY_RULES
from agent import STATE_ACTION_SIZE, run_population, trajectory_path
from plot import (
    handle_plotting_rei,
    handle_plotting_sana,
    populate_lists,
    unzipper_chungus_deluxe,
    plot_rules_followed_distribution
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

def parse_to_float(s):
    try:
        return float(s)
    except ValueError:
        try:
            numerator, denominator = map(float, s.split('/'))
            return numerator / denominator
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"Cannot convert '{s}' to float")
        
# def sample_from_database(num_pairs, database_path):
#     with open(database_path, "rb") as f:
#         database = pickle.load(f)
#     total_pairs = len(database)

#     if num_pairs == total_pairs:
#         return database_path
#     elif num_pairs < total_pairs:
#         new_pairs = random.sample(database, num_pairs)
#         new_database_path = trajectory_path + f"database_{num_pairs}.pkl"
#         with open(new_database_path, "wb") as f:
#             pickle.dump(new_pairs, f)
#         return new_database_path
#     else:
#         return -1


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
        "-c",
        "--composition",
        type=int,
        help="number of rules",
    )
    parse.add_argument(
        "-d",
        "--distribution",
        type=str,
        action="append",
        help="Distribution of segments collected",
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

    if args.composition and args.trajectories:
        rules.NUMBER_OF_RULES = args.composition
        database_path = f"trajectories/database_{args.trajectories[0]}_pairs_{args.composition}_rules.pkl"
    else:
        print("Missing either -c flag or -t flag")


    # database_path = ""
    # if args.trajectories and args.database:
    #     database_path = sample_from_database(args.trajectories[0], args.database)
    #     if database_path == -1:
    #         print("Provide a larger database, or generate a new one!")
    #         sys.exit(1)
    #     num_pairs = database_path.split("_")[1].split(".")[0]
    #     args.trajectories[0] = num_pairs
    # elif args.trajectories:
    #     database_path = f"trajectories/database_{args.trajectories[0]}.pkl"
    # elif args.database:
    #     database_path = args.database
    #     num_pairs = database_path.split("_")[1].split(".")[0]
    #     args.trajectories[0] = num_pairs
    # else:
    #     print(
    #         "Need to either provide number of trajectories to collect or existing database"
    #     )

    if args.distribution:
        try:
            rules.SEGMENT_DISTRIBUTION_BY_RULES = [parse_to_float(d) for d in args.distribution]
        except:
            print("Distribution input too advanced for Alex and Franklin's caveman parser. (or maybe you input something weird sry)")
            sys.exit()
        sum_dist = sum(rules.SEGMENT_DISTRIBUTION_BY_RULES)
        rules.SEGMENT_DISTRIBUTION_BY_RULES = [d / sum_dist for d in rules.SEGMENT_DISTRIBUTION_BY_RULES]
        assert (len(rules.SEGMENT_DISTRIBUTION_BY_RULES) == rules.NUMBER_OF_RULES + 1), f"SEGMENT_DISTRIBUTION_BY_RULES: {rules.SEGMENT_DISTRIBUTION_BY_RULES} does not have one more than the length specified in NUMBER_OF_RULES: {rules.NUMBER_OF_RULES}"
        assert (sum(rules.SEGMENT_DISTRIBUTION_BY_RULES) == 1), f"SEGMENT_DISTRIBUTION_BY_RULES: {rules.SEGMENT_DISTRIBUTION_BY_RULES} does not sum to 1 (even after scaling)"

    model_weights = ""
    if args.reward is None:
        # start the simulation in data collecting mode
        num_traj, collecting_rules_followed = start_simulation(
            "./config/data_collection_config.txt",
            args.trajectories[0],
            args.trajectories[0],
            "collect",
            args.headless,
        )

        print("Starting training on trajectories...")
        train_reward_function(database_path, args.epochs[0], args.parameters)

        # plot_rules_followed_distribution(
        #     collecting_rules_followed, "Input Distribution Rules Followed"
        # )

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
    truePairs, true_rules_followed = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trueRF",
        args.headless,
    )
    # plot_rules_followed_distribution(true_rules_followed, "Ground Truth Rules Followed")
    # plot_rules_followed_distribution(
        # true_rules_followed[-1000:], "Expert Ground Truth Rules Followed"
    # )

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
    # plot_rules_followed_distribution(
    #     trained_rules_followed, "Trained Agent Rules Followed"
    # )
    # plot_rules_followed_distribution(
    #     trained_rules_followed[-1000:], "Expert Trained Agent Rules Followed"
    # )

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
        true_agent_rewards,
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
    handle_plotting_rei(
        model_info,
        true_agent_expert_segments,
        true_agent_rewards,
        trained_agent_expert_segments,
        trained_agent_rewards,
        trained_segment_rules_satisifed,
        trained_segment_rewards,
        trained_segment_distances,
        training_segment_rules_satisfied,
        training_segment_rewards,
        training_segment_distances,
    )

    # num_rules = NUMBER_OF_RULES
    # best_true_agent_expert_segments, aggregate_trained_agent_expert_segments = unzipper_chungus_deluxe(num_rules)

    # handle_plotting_sana(
    #     best_true_agent_expert_segments,
    #     aggregate_trained_agent_expert_segments,
    # )
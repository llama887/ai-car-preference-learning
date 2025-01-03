import argparse
import glob
import os
import sys

import torch
import yaml

import agent
import reward
import rules

from rules import NUMBER_OF_RULES, SEGMENT_DISTRIBUTION_BY_RULES
from agent import STATE_ACTION_SIZE, run_population, trajectory_path, load_models
from plot import (
    handle_plotting_rei,
    handle_plotting_sana,
    populate_lists,
    unzipper_chungus_deluxe,
    plot_rules_followed_distribution,
)
from reward import (
    TrajectoryRewardNet,
    train_reward_function,
)

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


def parse_to_float(s):
    try:
        return float(s)
    except ValueError:
        try:
            numerator, denominator = map(float, s.split("/"))
            return numerator / denominator
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"Cannot convert '{s}' to float")


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
        "-s",
        "--segment",
        type=int,
        help="Length of segments",
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
        "--ensemble", action="store_true", help="Train an ensemble of 3 predictors"
    )
    parse.add_argument(
        "-r",
        "--reward",
        type=str,
        action="append",
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

    if args.distribution:
        try:
            rules.SEGMENT_DISTRIBUTION_BY_RULES = [
                parse_to_float(d) for d in args.distribution
            ]
        except:
            print(
                "Distribution input too advanced for Alex and Franklin's caveman parser. (or maybe you input something weird sry)"
            )
            sys.exit()
        sum_dist = sum(rules.SEGMENT_DISTRIBUTION_BY_RULES)
        rules.SEGMENT_DISTRIBUTION_BY_RULES = [
            d / sum_dist for d in rules.SEGMENT_DISTRIBUTION_BY_RULES
        ]
        assert (
            len(rules.SEGMENT_DISTRIBUTION_BY_RULES) == rules.NUMBER_OF_RULES + 1
        ), f"SEGMENT_DISTRIBUTION_BY_RULES: {rules.SEGMENT_DISTRIBUTION_BY_RULES} does not have one more than the length specified in NUMBER_OF_RULES: {rules.NUMBER_OF_RULES}"
        assert (
            sum(rules.SEGMENT_DISTRIBUTION_BY_RULES) == 1
        ), f"SEGMENT_DISTRIBUTION_BY_RULES: {rules.SEGMENT_DISTRIBUTION_BY_RULES} does not sum to 1 (even after scaling)"


    if args.segment and args.segment < 1:
        raise Exception("Can not have segments with length < 1")
    agent.train_trajectory_length = args.segment if args.segment else 1

    model_weights = ""
    if args.reward is None:
        # start the simulation in data collecting mode
        num_traj, collecting_rules_followed = start_simulation(
            "./config/data_collection_config.txt",
            args.trajectories[0],
            args.trajectories[0],
            "collect",
            args.headless,
            args.ensemble,
        )

        print("Starting training on trajectories...")
        train_reward_function(
            database_path, args.epochs[0], args.parameters, args.ensemble
        )

        print("Finished training model...")

        if not args.parameters:
            sys.exit()
        # run the simulation with the trained reward function
        if args.ensemble:
            model_weights = ["QUICK", reward.ensemble_path]
        else:
            model_weights = [(reward.models_path + f"model_{args.epochs[0]}.pth")]
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
        args.ensemble,
    )

    with open(
        args.parameters if args.parameters is not None else "best_params.yaml", "r"
    ) as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]

    print("Simulating on trained reward function...")
    # agent.reward_network = TrajectoryRewardNet(
    #     STATE_ACTION_SIZE * 2, hidden_size=hidden_size
    # ).to(device)

    # weights = torch.load(model_weights, map_location=device)
    # agent.reward_network.load_state_dict(weights)
    load_models(model_weights, hidden_size)

    trainedPairs, trained_rules_followed = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trainedRF",
        args.headless,
        args.ensemble,
    )

    model_info = {
        "net": agent.reward_network,
        "ensemble": agent.ensemble,
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

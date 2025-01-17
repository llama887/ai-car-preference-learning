import argparse
import os
import sys

import yaml

import agent
import reward
import rules

from rules import NUMBER_OF_RULES, SEGMENT_DISTRIBUTION_BY_RULES
from agent import STATE_ACTION_SIZE, AGENTS_PER_GENERATION, run_population, load_models

os.environ["WANDB_SILENT"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
        "-g",
        "--generations",
        type=int,
        nargs=1,
        help="Number of generations to train the agent",
    )
    parse.add_argument(
        "--headless", action="store_true", help="Run simulation without GUI"
    )
    parse.add_argument(
        "-c",
        "--composition",
        type=int,
        help="number of rules",
    )

    args = parse.parse_args()

    if args.generations is not None and args.generations[0] < 0:
        raise Exception("Invalid input. All arguments must be positive integers.")

    # Display simulator or not
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Set distribution of segment rule satisfaction
    if args.composition:
        rules.NUMBER_OF_RULES = args.composition
    else:
        raise Exception("Missing -c flag")

    # run the simulation with the true reward function (if trajectories do not exist yet)
    if os.path.exists(f"trueRF_trajectories/trueRF_{args.generations * AGENTS_PER_GENERATION}_trajectories_{rules.NUMBER_OF_RULES}_rules.pkl"):
        truePairs = args.generations * AGENTS_PER_GENERATION
        print(f"trueRF already exists with {truePairs} trajectories on {rules.NUMBER_OF_RULES} rules")
    else:
        print("Simulating on true reward function...")
        truePairs, true_rules_followed = start_simulation(
            "./config/agent_config.txt",
            args.generations[0],
            0,
            "trueRF",
            args.headless,
            False,
        )

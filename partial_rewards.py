import argparse
import os
import sys
import agent
import numpy as np
import rules
from agent import run_population
from reward import (
    train_reward_function,
)
import plotly.figure_factory as ff

rules.PARTIAL_REWARD = True
rules.NUMBER_OF_RULES = 2

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
        "-r",
        "--resolution",
        type=int,
        help="Resolution of gridding",
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
        or (args.generations is not None and args.generations[0] < 0)
        or (args.epochs is not None and args.epochs[0] < 0)
    ):
        print("Invalid input. All arguments must be positive integers.")
        sys.exit(1)

    # Display simulator or not
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    # Set distribution of segment rule satisfaction
    if args.trajectories and args.parameters:
        num_pairs = args.trajectories[0]
        database_path = f"trajectories/database_{num_pairs}_pairs_{args.composition}_rules_{agent.train_trajectory_length}_length.pkl"
    else:
        print("Missing either -p flag or -t flag")
        sys.exit()

    if args.resolution < 1:
        raise Exception("Resolution must be greater than 1")

    grid_points = 1
    distributions = [[] for _ in range(2)]
    val_accs = []
    step = 1 / args.resolution
    for a in np.arange(0, 1 + step, step):
        for b in np.arange(0, 1 - a + step, step):
            c = 1 - a - b
            if c >= 0:
                distributions[0].append(a)
                distributions[1].append(b)
                distributions[2].append(c)
                rules.SEGMENT_DISTRIBUTION_BY_RULES = [a, b, c]

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
                val_accs.append(train_reward_function(
                    database_path, args.epochs[0], args.parameters, args.ensemble, None, "val_acc"
                ))
                print("Finished training model...")
            grid_points += 1
    fig = ff.create_ternary_contour(np.array(distributions), val_accs,
                                pole_labels=['0 Rule', '1 Rule', '2 Rule'],
                                interp_mode='cartesian',
                                showscale=True,
                                title=dict(
                                  text='Final Validation for Various Database Distributions'
                                ))
    fig.write_image("figures/simplex.png")

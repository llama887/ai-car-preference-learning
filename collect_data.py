import argparse
import sys

import agent
import rules
from main import parse_to_float, start_simulation

if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
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
        help="Length of segments to collect",
    )
    parse.add_argument(
        "--headless", action="store_true", help="Run simulation without GUI"
    )
    parse.add_argument(
        "-db",
        "--database",
        type=str,
        nargs=1,
        help="Path to master database",
    )
    parse.add_argument(
        "-tp",
        "--trajectory_path",
        type=str,
        nargs=1,
        help="Path to save trajectory pkl files",
    )
    parse.add_argument(
        "-p",
        "--paired",
        action="store_true",
        help="flag to generate database already paired"
    )
    args = parse.parse_args()

    if args.segment and args.segment < 1:
        raise Exception("Can not have segments with length < 1")
    agent.train_trajectory_length = args.segment if args.segment else 1

    rules.NUMBER_OF_RULES = 3
    rules.RULES_INCLUDED = [1, 2, 3]
    rules.SEGMENT_DISTRIBUTION_BY_RULES = [1/6, 1/6, 1/6, 1/2]
    
    if args.database:
        if args.paired:
            agent.paired_database = args.database[0]
        agent.master_database = args.database[0]
    if args.trajectory_path:
        agent.trajectories_path = args.trajectory_path[0]
    if args.trajectories is not None and args.trajectories[0] > 0:
        num_traj, collecting_rules_followed = start_simulation(
            "./config/data_collection_config.txt",
            args.trajectories[0],
            args.trajectories[0],
            "collect",
            args.headless,
        )

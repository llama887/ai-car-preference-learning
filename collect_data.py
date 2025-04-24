import argparse
import sys

import agent
import rules
from main import parse_to_float, start_simulation
from itertools import combinations
import random

rule_range = [i for i in range(1, rules.TOTAL_RULES + 1)]
buckets = [list(combinations(rule_range, r)) for r in range(1, len(rule_range) + 1)]  
buckets = [list(sublist) for g in buckets for sublist in g]  

uniform_weights = [1 / len(buckets) for _ in range(len(buckets))]

sum_rules = sum([len(bucket) for bucket in buckets]) # scaled by number of rules
adjusted_weights = [len(bucket) / sum_rules for bucket in buckets]


def set_reward():
    rules.RULES_INCLUDED = random.choices(buckets, weights=adjusted_weights, k=1)[0]
    rules.NUMBER_OF_RULES = len(rules.RULES_INCLUDED)
    rules.SEGMENT_DISTRIBUTION_BY_RULES = [1/(2 * rules.NUMBER_OF_RULES) for _ in range(rules.NUMBER_OF_RULES)] + [1/2]

    print("THIS PROCESS IS USING THE FOLLOWING RULES:")
    print(rules.RULES_INCLUDED)
    print("THIS PROCESS IS USING THE FOLLOWING SEGMENT DISTRIBUTION:")
    print(rules.SEGMENT_DISTRIBUTION_BY_RULES)

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

    set_reward()
    
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

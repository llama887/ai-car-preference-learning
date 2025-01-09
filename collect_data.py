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
        "--headless", action="store_true", help="Run simulation without GUI"
    )
    parse.add_argument(
        "-d",
        "--distribution",
        type=str,
        action="append",
        help="Distribution of segments collected",
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

    args = parse.parse_args()

    if args.distribution:
        try:
            rules.SEGMENT_DISTRIBUTION_BY_RULES = [
                parse_to_float(d) for d in args.distribution
            ]
        except Exception:
            print(
                "Distribution input too advanced for Alex and Franklin's caveman parser. (or maybe you input something weird sry)"
            )
            sys.exit()
        sum_dist = sum(rules.SEGMENT_DISTRIBUTION_BY_RULES)
        rules.SEGMENT_DISTRIBUTION_BY_RULES = [
            d / sum_dist for d in rules.SEGMENT_DISTRIBUTION_BY_RULES
        ]
        rules.NUMBER_OF_RULES = len(rules.SEGMENT_DISTRIBUTION_BY_RULES) - 1
        assert (
            sum(rules.SEGMENT_DISTRIBUTION_BY_RULES) == 1
        ), f"SEGMENT_DISTRIBUTION_BY_RULES: {rules.SEGMENT_DISTRIBUTION_BY_RULES} does not sum to 1 (even after scaling)"
    if args.database:
        agent.master_database = args.database[0]
    if args.trajectory_path:
        agent.trajectory_path = args.trajectory_path[0]
    if args.trajectories is not None and args.trajectories[0] > 0:
        num_traj, collecting_rules_followed = start_simulation(
            "./config/data_collection_config.txt",
            args.trajectories[0],
            args.trajectories[0],
            "collect",
            args.headless,
        )

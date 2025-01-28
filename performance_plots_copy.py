import argparse
import glob
import os
import pickle
import re
import shutil
import zipfile

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import reward
from agent import AGENTS_PER_GENERATION

zips_path = "zips/"


def unzipper_chungus_deluxe(num_rules):
    best_true_agent_satisfaction_segments = {}
    aggregate_trained_agent_satisfaction_segments = {}
    for rule_count in range(1, num_rules + 1):
        rule_best_true_agent_segments = [[0]]
        rule_aggregate_segments = {}

        zip_files = glob.glob(f"{zips_path}trajectories_t*_r{rule_count}.zip")
        print(f"{rule_count} rule files:", zip_files)
        for zip_file in zip_files:
            if not zip_file:
                raise Exception("Zip files missing")
            num_pairs = int(re.search(r"trajectories_t(\d+)*", zip_file).group(1))
            true_agent_satisfaction_segments = []
            trained_agent_satisfaction_segments = []

            os.makedirs("temp_trajectories", exist_ok=True)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                # Extract all contents of the zip file to the specified folder
                zip_ref.extractall("temp_trajectories")
                trainedRF = glob.glob(
                    "temp_trajectories/trajectories*/trainedRF_*.pkl"
                )[0]

                with open(trainedRF, "rb") as f:
                    trained_trajectories = pickle.load(f)

                num_trajectories = len(trained_trajectories)
                trueRF = glob.glob(
                    f"trueRF_trajectories/trueRF_{num_trajectories}_trajectories_{num_rules}_rules.pkl"
                )[0]
                with open(trueRF, "rb") as f:
                    true_trajectories = pickle.load(f)

                num_true_trajectories = len(trained_trajectories)
                count = 0
                while count < num_true_trajectories:
                    gen_true_satisfaction_segments = []
                    for _ in range(AGENTS_PER_GENERATION):
                        trajectory = true_trajectories[count]
                        gen_true_satisfaction_segments.append(
                            trajectory.num_satisfaction_segments
                        )
                        count += 1
                    if gen_true_satisfaction_segments:
                        true_agent_satisfaction_segments.append(
                            gen_true_satisfaction_segments
                        )

                num_trained_trajectories = len(trained_trajectories)
                count = 0
                while count < num_trained_trajectories:
                    gen_trained_satisfaction_segments = []
                    for _ in range(AGENTS_PER_GENERATION):
                        trajectory = trained_trajectories[count]
                        gen_trained_satisfaction_segments.append(
                            trajectory.num_expert_segments
                        )
                        count += 1
                    if gen_trained_satisfaction_segments:
                        trained_agent_satisfaction_segments.append(
                            gen_trained_satisfaction_segments
                        )

                if max(
                    [sum(generation) for generation in true_agent_satisfaction_segments]
                ) > max(
                    [sum(generation) for generation in rule_best_true_agent_segments]
                ):
                    rule_best_true_agent_segments = (
                        true_agent_satisfaction_segments.copy()
                    )
                rule_aggregate_segments[num_pairs] = (
                    trained_agent_satisfaction_segments.copy()
                )

            shutil.rmtree("temp_trajectories")
        best_true_agent_satisfaction_segments[rule_count] = (
            rule_best_true_agent_segments.copy()
        )
        aggregate_trained_agent_satisfaction_segments[rule_count] = (
            rule_aggregate_segments
        )

    return (
        best_true_agent_satisfaction_segments,
        aggregate_trained_agent_satisfaction_segments,
    )


def handle_plotting_sana(
    best_true_agent_satisfaction_segments, aggregate_trained_agent_satisfaction_segments
):
    # Avg/max number of satisfaction segments per trajectory for each agent over generations
    best_trueRF_average_satisfaction_segments = {}
    best_trueRF_max_satisfaction_segments = {}

    for rule in best_true_agent_satisfaction_segments.keys():
        best_trueRF_average_satisfaction_segments[rule] = [
            (sum(generation) / AGENTS_PER_GENERATION)
            for generation in best_true_agent_satisfaction_segments[rule]
        ]
        best_trueRF_max_satisfaction_segments[rule] = [
            max(generation)
            for generation in best_true_agent_satisfaction_segments[rule]
        ]

    # Avg/max number of satisfaction segments per trajectory for each agent over generations
    aggregate_trainedRF_average_satisfaction_segments = {}
    aggregate_trainedRF_max_satisfaction_segments = {}
    for rule in aggregate_trained_agent_satisfaction_segments.keys():
        aggregate_trainedRF_average_satisfaction_segments[rule] = {}
        aggregate_trainedRF_max_satisfaction_segments[rule] = {}
        for num_pairs in aggregate_trained_agent_satisfaction_segments[rule].keys():
            aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs] = [
                (sum(generation) / AGENTS_PER_GENERATION)
                for generation in aggregate_trained_agent_satisfaction_segments[rule][
                    num_pairs
                ]
            ]
            aggregate_trainedRF_max_satisfaction_segments[rule][num_pairs] = [
                max(generation)
                for generation in aggregate_trained_agent_satisfaction_segments[rule][
                    num_pairs
                ]
            ]

    graph_normalized_segments_over_generations(
        best_trueRF_average_satisfaction_segments,
        aggregate_trainedRF_average_satisfaction_segments,
        best_trueRF_max_satisfaction_segments,
        aggregate_trainedRF_max_satisfaction_segments,
    )

    graph_gap_over_pairs(
        best_trueRF_average_satisfaction_segments,
        aggregate_trainedRF_average_satisfaction_segments,
    )


def graph_normalized_segments_over_generations(
    best_trueRF_average_satisfaction_segments,
    aggregate_trainedRF_average_satisfaction_segments,
    best_trueRF_max_satisfaction_segments,
    aggregate_trainedRF_max_satisfaction_segments,
):
    os.makedirs(reward.figure_path, exist_ok=True)

    for rule in best_trueRF_average_satisfaction_segments.keys():
        x_values = range(len(best_trueRF_average_satisfaction_segments[rule]))
        max_avg_trueRF_segments = max(best_trueRF_average_satisfaction_segments[rule])
        max_max_trueRF_segments = max(best_trueRF_max_satisfaction_segments[rule])
        best_trueRF_average_satisfaction_segments[rule] = [
            avg_segments / max_avg_trueRF_segments
            for avg_segments in best_trueRF_average_satisfaction_segments[rule]
        ]
        best_trueRF_max_satisfaction_segments[rule] = [
            max_segments / max_max_trueRF_segments
            for max_segments in best_trueRF_max_satisfaction_segments[rule]
        ]

        for num_pairs in aggregate_trainedRF_average_satisfaction_segments[rule].keys():
            aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs] = [
                avg_segments / max_avg_trueRF_segments
                for avg_segments in aggregate_trainedRF_average_satisfaction_segments[
                    rule
                ][num_pairs]
            ]
        for num_pairs in aggregate_trainedRF_max_satisfaction_segments[rule].keys():
            aggregate_trainedRF_max_satisfaction_segments[rule][num_pairs] = [
                max_segments / max_max_trueRF_segments
                for max_segments in aggregate_trainedRF_max_satisfaction_segments[rule][
                    num_pairs
                ]
            ]

        plt.figure()
        plt.plot(
            x_values,
            best_trueRF_average_satisfaction_segments[rule],
            label="Ground Truth",
        )
        for num_pairs in sorted(
            aggregate_trainedRF_average_satisfaction_segments[rule].keys()
        ):
            plt.plot(
                x_values,
                aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs],
                label=f"{num_pairs} pairs",
            )
        plt.xlabel("Generation")
        plt.ylabel("Ground Truth Reward (wrt GT Agent)")
        plt.legend()
        plt.savefig(f"{reward.figure_path}average_norm_{rule}_rules.png")
        plt.close()

        plt.figure()
        plt.plot(
            x_values, best_trueRF_max_satisfaction_segments[rule], label="Ground Truth"
        )
        for num_pairs in aggregate_trainedRF_max_satisfaction_segments[rule].keys():
            plt.plot(
                x_values,
                aggregate_trainedRF_max_satisfaction_segments[rule][num_pairs],
                label=f"{num_pairs} pairs",
            )
        plt.xlabel("Generation")
        plt.ylabel("Ground Truth Reward (wrt GT Agent)")
        plt.legend()
        plt.savefig(f"{reward.figure_path}max_norm_{rule}_rules.png")
        plt.close()


def graph_gap_over_pairs(
    best_trueRF_average_satisfaction_segments,
    aggregate_trainedRF_average_satisfaction_segments,
):
    os.makedirs(reward.figure_path, exist_ok=True)
    plt.figure()
    for rule in best_trueRF_average_satisfaction_segments.keys():
        x_values = sorted(
            aggregate_trainedRF_average_satisfaction_segments[rule].keys()
        )
        y_values = []

        max_avg_trueRF_segments = max(best_trueRF_average_satisfaction_segments[rule])
        best_trueRF_average_satisfaction_segments[rule] = [
            avg_segments / max_avg_trueRF_segments
            for avg_segments in best_trueRF_average_satisfaction_segments[rule]
        ]
        for num_pairs in aggregate_trainedRF_average_satisfaction_segments[rule].keys():
            aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs] = [
                avg_segments / max_avg_trueRF_segments
                for avg_segments in aggregate_trainedRF_average_satisfaction_segments[
                    rule
                ][num_pairs]
            ]

        for num_pairs in x_values:
            if len(best_trueRF_average_satisfaction_segments[rule]) != len(
                aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs]
            ):
                print("Mismatch in generations between GT and Trained!")
            gap = sum(
                [
                    best_trueRF_average_satisfaction_segments[rule][generation]
                    - aggregate_trainedRF_average_satisfaction_segments[rule][
                        num_pairs
                    ][generation]
                    for generation in range(
                        len(best_trueRF_average_satisfaction_segments[rule])
                    )
                ]
            )
            y_values.append(gap / len(best_trueRF_average_satisfaction_segments[rule]))
        plt.plot(x_values, y_values, label=f"{rule} rules")
    plt.xlabel("Number of Trajectory Pairs (Log Scale)")
    plt.xscale("log")
    plt.ylabel("Average Gap in Reward (Reward_GT - Reward_Trained)")
    plt.legend()
    plt.savefig(f"{reward.figure_path}gap.png")
    plt.close()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Performance plots")
    parse.add_argument(
        "-c",
        "--composition",
        type=int,
        help="number of rules",
    )
    args = parse.parse_args()
    num_rules = args.composition
    (
        best_true_agent_satisfaction_segments,
        aggregate_trained_agent_satisfaction_segments,
    ) = unzipper_chungus_deluxe(num_rules)

    handle_plotting_sana(
        best_true_agent_satisfaction_segments,
        aggregate_trained_agent_satisfaction_segments,
    )

zips_path = "zips/"


def unzipper_chungus_deluxe(num_rules):
    best_true_agent_satisfaction_segments = {}
    aggregate_trained_agent_satisfaction_segments = {}
    for rule_count in range(1, num_rules + 1):
        rule_best_true_agent_segments = [[0]]
        rule_aggregate_segments = {}

        zip_files = glob.glob(f"{zips_path}trajectories_t*_r{rule_count}.zip")
        print(f"{rule_count} rule files:", zip_files)
        for zip_file in zip_files:
            if not zip_file:
                raise Exception("Zip files missing")
            num_pairs = int(re.search(r"trajectories_t(\d+)*", zip_file).group(1))
            true_agent_satisfaction_segments = []
            trained_agent_satisfaction_segments = []

            os.makedirs("temp_trajectories", exist_ok=True)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                # Extract all contents of the zip file to the specified folder
                zip_ref.extractall("temp_trajectories")
            trainedRF = glob.glob("temp_trajectories/trajectories*/trainedRF_*.pkl")[0]

            with open(trainedRF, "rb") as f:
                trained_trajectories = pickle.load(f)

            num_trajectories = len(trained_trajectories)
            trueRF = glob.glob(
                f"trueRF_trajectories/trueRF_{num_trajectories}_trajectories_{num_rules}_rules.pkl"
            )[0]
            with open(trueRF, "rb") as f:
                true_trajectories = pickle.load(f)

            num_true_trajectories = len(trained_trajectories)
            count = 0
            while count < num_true_trajectories:
                gen_true_satisfaction_segments = []
                for i in range(AGENTS_PER_GENERATION):
                    trajectory = true_trajectories[count]
                    gen_true_satisfaction_segments.append(
                        trajectory.num_satisfaction_segments
                    )
                    count += 1
                if gen_true_satisfaction_segments:
                    true_agent_satisfaction_segments.append(
                        gen_true_satisfaction_segments
                    )

            num_trained_trajectories = len(trained_trajectories)
            count = 0
            while count < num_trained_trajectories:
                gen_trained_satisfaction_segments = []
                for i in range(AGENTS_PER_GENERATION):
                    trajectory = trained_trajectories[count]
                    gen_trained_satisfaction_segments.append(
                        trajectory.num_expert_segments
                    )
                    count += 1
                if gen_trained_satisfaction_segments:
                    trained_agent_satisfaction_segments.append(
                        gen_trained_satisfaction_segments
                    )

            if max(
                [sum(generation) for generation in true_agent_satisfaction_segments]
            ) > max([sum(generation) for generation in rule_best_true_agent_segments]):
                rule_best_true_agent_segments = true_agent_satisfaction_segments.copy()
            rule_aggregate_segments[num_pairs] = (
                trained_agent_satisfaction_segments.copy()
            )

            shutil.rmtree("temp_trajectories")
        best_true_agent_satisfaction_segments[rule_count] = (
            rule_best_true_agent_segments.copy()
        )
        aggregate_trained_agent_satisfaction_segments[rule_count] = (
            rule_aggregate_segments
        )

    return (
        best_true_agent_satisfaction_segments,
        aggregate_trained_agent_satisfaction_segments,
    )


def handle_plotting_sana(
    best_true_agent_satisfaction_segments, aggregate_trained_agent_satisfaction_segments
):
    # Avg/max number of satisfaction segments per trajectory for each agent over generations
    best_trueRF_average_satisfaction_segments = {}
    best_trueRF_max_satisfaction_segments = {}

    for rule in best_true_agent_satisfaction_segments.keys():
        best_trueRF_average_satisfaction_segments[rule] = [
            (sum(generation) / AGENTS_PER_GENERATION)
            for generation in best_true_agent_satisfaction_segments[rule]
        ]
        best_trueRF_max_satisfaction_segments[rule] = [
            max(generation)
            for generation in best_true_agent_satisfaction_segments[rule]
        ]

    # Avg/max number of satisfaction segments per trajectory for each agent over generations
    aggregate_trainedRF_average_satisfaction_segments = {}
    aggregate_trainedRF_max_satisfaction_segments = {}
    for rule in aggregate_trained_agent_satisfaction_segments.keys():
        aggregate_trainedRF_average_satisfaction_segments[rule] = {}
        aggregate_trainedRF_max_satisfaction_segments[rule] = {}
        for num_pairs in aggregate_trained_agent_satisfaction_segments[rule].keys():
            aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs] = [
                (sum(generation) / AGENTS_PER_GENERATION)
                for generation in aggregate_trained_agent_satisfaction_segments[rule][
                    num_pairs
                ]
            ]
            aggregate_trainedRF_max_satisfaction_segments[rule][num_pairs] = [
                max(generation)
                for generation in aggregate_trained_agent_satisfaction_segments[rule][
                    num_pairs
                ]
            ]

    graph_normalized_segments_over_generations(
        best_trueRF_average_satisfaction_segments,
        aggregate_trainedRF_average_satisfaction_segments,
        best_trueRF_max_satisfaction_segments,
        aggregate_trainedRF_max_satisfaction_segments,
    )

    graph_gap_over_pairs(
        best_trueRF_average_satisfaction_segments,
        aggregate_trainedRF_average_satisfaction_segments,
    )


def graph_normalized_segments_over_generations(
    best_trueRF_average_satisfaction_segments,
    aggregate_trainedRF_average_satisfaction_segments,
    best_trueRF_max_satisfaction_segments,
    aggregate_trainedRF_max_satisfaction_segments,
):
    os.makedirs(reward.figure_path, exist_ok=True)

    for rule in best_trueRF_average_satisfaction_segments.keys():
        x_values = range(len(best_trueRF_average_satisfaction_segments[rule]))
        max_avg_trueRF_segments = max(best_trueRF_average_satisfaction_segments[rule])
        max_max_trueRF_segments = max(best_trueRF_max_satisfaction_segments[rule])
        best_trueRF_average_satisfaction_segments[rule] = [
            avg_segments / max_avg_trueRF_segments
            for avg_segments in best_trueRF_average_satisfaction_segments[rule]
        ]
        best_trueRF_max_satisfaction_segments[rule] = [
            max_segments / max_max_trueRF_segments
            for max_segments in best_trueRF_max_satisfaction_segments[rule]
        ]

        for num_pairs in aggregate_trainedRF_average_satisfaction_segments[rule].keys():
            aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs] = [
                avg_segments / max_avg_trueRF_segments
                for avg_segments in aggregate_trainedRF_average_satisfaction_segments[
                    rule
                ][num_pairs]
            ]
        for num_pairs in aggregate_trainedRF_max_satisfaction_segments[rule].keys():
            aggregate_trainedRF_max_satisfaction_segments[rule][num_pairs] = [
                max_segments / max_max_trueRF_segments
                for max_segments in aggregate_trainedRF_max_satisfaction_segments[rule][
                    num_pairs
                ]
            ]

        plt.figure()
        plt.plot(
            x_values,
            best_trueRF_average_satisfaction_segments[rule],
            label="Ground Truth",
        )
        for num_pairs in sorted(
            aggregate_trainedRF_average_satisfaction_segments[rule].keys()
        ):
            plt.plot(
                x_values,
                aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs],
                label=f"{num_pairs} pairs",
            )
        plt.xlabel("Generation")
        plt.ylabel("Ground Truth Reward (wrt GT Agent)")
        plt.legend()
        plt.savefig(f"{reward.figure_path}average_norm_{rule}_rules.png")
        plt.close()

        plt.figure()
        plt.plot(
            x_values, best_trueRF_max_satisfaction_segments[rule], label="Ground Truth"
        )
        for num_pairs in aggregate_trainedRF_max_satisfaction_segments[rule].keys():
            plt.plot(
                x_values,
                aggregate_trainedRF_max_satisfaction_segments[rule][num_pairs],
                label=f"{num_pairs} pairs",
            )
        plt.xlabel("Generation")
        plt.ylabel("Ground Truth Reward (wrt GT Agent)")
        plt.legend()
        plt.savefig(f"{reward.figure_path}max_norm_{rule}_rules.png")
        plt.close()


def graph_gap_over_pairs(
    best_trueRF_average_satisfaction_segments,
    aggregate_trainedRF_average_satisfaction_segments,
):
    os.makedirs(reward.figure_path, exist_ok=True)
    plt.figure()
    for rule in best_trueRF_average_satisfaction_segments.keys():
        x_values = sorted(
            aggregate_trainedRF_average_satisfaction_segments[rule].keys()
        )
        y_values = []

        max_avg_trueRF_segments = max(best_trueRF_average_satisfaction_segments[rule])
        best_trueRF_average_satisfaction_segments[rule] = [
            avg_segments / max_avg_trueRF_segments
            for avg_segments in best_trueRF_average_satisfaction_segments[rule]
        ]
        for num_pairs in aggregate_trainedRF_average_satisfaction_segments[rule].keys():
            aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs] = [
                avg_segments / max_avg_trueRF_segments
                for avg_segments in aggregate_trainedRF_average_satisfaction_segments[
                    rule
                ][num_pairs]
            ]

        for num_pairs in x_values:
            if len(best_trueRF_average_satisfaction_segments[rule]) != len(
                aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs]
            ):
                print("Mismatch in generations between GT and Trained!")
            gap = sum(
                [
                    best_trueRF_average_satisfaction_segments[rule][generation]
                    - aggregate_trainedRF_average_satisfaction_segments[rule][
                        num_pairs
                    ][generation]
                    for generation in range(
                        len(best_trueRF_average_satisfaction_segments[rule])
                    )
                ]
            )
            y_values.append(gap / len(best_trueRF_average_satisfaction_segments[rule]))
        plt.plot(x_values, y_values, label=f"{rule} rules")
    plt.xlabel("Number of Trajectory Pairs (Log Scale)")
    plt.xscale("log")
    plt.ylabel("Average Gap in Reward (Reward_GT - Reward_Trained)")
    plt.legend()
    plt.savefig(f"{reward.figure_path}gap.png")
    plt.close()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Performance plots")
    parse.add_argument(
        "-c",
        "--composition",
        type=int,
        help="number of rules",
    )
    args = parse.parse_args()
    num_rules = args.composition
    (
        best_true_agent_satisfaction_segments,
        aggregate_trained_agent_satisfaction_segments,
    ) = unzipper_chungus_deluxe(num_rules)

    handle_plotting_sana(
        best_true_agent_satisfaction_segments,
        aggregate_trained_agent_satisfaction_segments,
    )

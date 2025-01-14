import os
import pickle
import matplotlib.pyplot as plt
import argparse

import reward
import glob
import shutil
import zipfile
import re

zips_path = "zips/"

AGENTS_PER_GENERATION = 20
def unzipper_chungus(num_rules):
    best_true_agent_expert_segments = [[0]]
    aggregate_trained_agent_expert_segments = {}

    zip_files = glob.glob(f"{zips_path}trajectories_t*_r{num_rules}.zip")
    for zip_file in zip_files:
        if not zip_file:
            raise Exception("Zip files missing")
        num_pairs = int(re.search(r"trajectories_(\d+)_pairs", zip_file).group(1))

        true_agent_expert_segments = []
        trained_agent_expert_segments = []
        os.makedirs("temp_trajectories", exist_ok=True)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            # Extract all contents of the zip file to the specified folder
            zip_ref.extractall("temp_trajectories")
            trueRF = glob.glob(f"temp_trajectories/trajectories/trueRF_*.pkl")[0]
            trainedRF = glob.glob(f"temp_trajectories/trajectories/trainedRF_*.pkl")[0]

            with open(trueRF, "rb") as f:
                true_trajectories = pickle.load(f)
            with open(trainedRF, "rb") as f:
                trained_trajectories = pickle.load(f)

            num_true_trajectories = len(true_trajectories)
            count = 0
            while count < num_true_trajectories:
                gen_true_expert_segments = []
                for _ in range(AGENTS_PER_GENERATION // 2):
                    trajectory_pair = true_trajectories[count]
                    gen_true_expert_segments.extend(
                        [trajectory_pair.e1, trajectory_pair.e2]
                    )
                    count += 1
                if gen_true_expert_segments:
                    true_agent_expert_segments.append(gen_true_expert_segments)

            num_trained_trajectories = len(trained_trajectories)

            count = 0
            while count < num_trained_trajectories:
                gen_trained_expert_segments = []
                for _ in range(AGENTS_PER_GENERATION // 2):
                    trajectory_pair = trained_trajectories[count]
                    gen_trained_expert_segments.extend(
                        [trajectory_pair.e1, trajectory_pair.e2]
                    )
                    count += 1
                if gen_trained_expert_segments:
                    trained_agent_expert_segments.append(gen_trained_expert_segments)

            if max(
                [sum(generation) for generation in true_agent_expert_segments]
            ) > max(
                [sum(generation) for generation in best_true_agent_expert_segments]
            ):
                best_true_agent_expert_segments = true_agent_expert_segments.copy()
            aggregate_trained_agent_expert_segments[num_pairs] = (
                trained_agent_expert_segments.copy()
            )

        shutil.rmtree("temp_trajectories")
    return (
        best_true_agent_expert_segments,
        aggregate_trained_agent_expert_segments,
    )


def unzipper_chungus_deluxe(num_rules):
    best_true_agent_expert_segments = {}
    aggregate_trained_agent_expert_segments = {}
    for rule_count in range(1, num_rules + 1):
        rule_best_true_agent_segments = [[0]]
        rule_aggregate_segments = {}

        zip_files = glob.glob(f"{zips_path}trajectories_t*_r{rule_count}.zip")
        print(zip_files)
        for zip_file in zip_files:
            if not zip_file:
                raise Exception("Zip files missing")
            num_pairs = int(re.search(r"trajectories_t(\d+)*", zip_file).group(1))
            print(num_pairs)
            true_agent_expert_segments = []
            trained_agent_expert_segments = []

            os.makedirs("temp_trajectories", exist_ok=True)
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                # Extract all contents of the zip file to the specified folder
                zip_ref.extractall("temp_trajectories")
                trueRF = glob.glob(f"temp_trajectories/trajectories*/trueRF_*.pkl")[0]
                trainedRF = glob.glob(
                    f"temp_trajectories/trajectories*/trainedRF_*.pkl"
                )[0]

                with open(trueRF, "rb") as f:
                    true_trajectories = pickle.load(f)
                with open(trainedRF, "rb") as f:
                    trained_trajectories = pickle.load(f)

                num_true_trajectories = len(true_trajectories)
                count = 0
                while count < num_true_trajectories:
                    gen_true_expert_segments = []
                    for _ in range(AGENTS_PER_GENERATION):
                        trajectory = true_trajectories[count]
                        gen_true_expert_segments.append(trajectory.num_expert_segments)
                        count += 1
                    if gen_true_expert_segments:
                        true_agent_expert_segments.append(gen_true_expert_segments)

                num_trained_trajectories = len(trained_trajectories)
                count = 0
                while count < num_trained_trajectories:
                    gen_trained_expert_segments = []
                    for _ in range(AGENTS_PER_GENERATION):
                        trajectory = trained_trajectories[count]
                        gen_trained_expert_segments.append(
                            trajectory.num_expert_segments
                        )
                        count += 1
                    if gen_trained_expert_segments:
                        trained_agent_expert_segments.append(
                            gen_trained_expert_segments
                        )

                if max(
                    [sum(generation) for generation in true_agent_expert_segments]
                ) > max(
                    [sum(generation) for generation in rule_best_true_agent_segments]
                ):
                    rule_best_true_agent_segments = true_agent_expert_segments.copy()
                rule_aggregate_segments[num_pairs] = (
                    trained_agent_expert_segments.copy()
                )

            shutil.rmtree("temp_trajectories")
        best_true_agent_expert_segments[rule_count] = (
            rule_best_true_agent_segments.copy()
        )
        aggregate_trained_agent_expert_segments[rule_count] = rule_aggregate_segments

    # s = 0
    # for i in range(len(best_true_agent_expert_segments[1])):
    #     best_true_sum = sum(best_true_agent_expert_segments[1][i])
    #     trained_sum = sum(aggregate_trained_agent_expert_segments[1][1000000][i])
    #     diff = best_true_sum - trained_sum

    #     # Use f-strings to format with fixed width
    #     print(f"{i:<5} {best_true_sum:<15} {trained_sum:<15} {diff:<15} {s:<15}")
    #     s += diff

    return (
        best_true_agent_expert_segments,
        aggregate_trained_agent_expert_segments,
    )


def handle_plotting_sana(
    best_true_agent_expert_segments, aggregate_trained_agent_expert_segments
):
    # Avg/max number of expert segments per trajectory for each agent over generations
    best_trueRF_average_expert_segments = {}
    best_trueRF_max_expert_segments = {}

    for rule in best_true_agent_expert_segments.keys():
        best_trueRF_average_expert_segments[rule] = [
            (sum(generation) / AGENTS_PER_GENERATION)
            for generation in best_true_agent_expert_segments[rule]
        ]
        best_trueRF_max_expert_segments[rule] = [
            max(generation) for generation in best_true_agent_expert_segments[rule]
        ]

    # Avg/max number of expert segments per trajectory for each agent over generations
    aggregate_trainedRF_average_expert_segments = {}
    aggregate_trainedRF_max_expert_segments = {}
    for rule in aggregate_trained_agent_expert_segments.keys():
        aggregate_trainedRF_average_expert_segments[rule] = {}
        aggregate_trainedRF_max_expert_segments[rule] = {}
        for num_pairs in aggregate_trained_agent_expert_segments[rule].keys():
            aggregate_trainedRF_average_expert_segments[rule][num_pairs] = [
                (sum(generation) / AGENTS_PER_GENERATION)
                for generation in aggregate_trained_agent_expert_segments[rule][
                    num_pairs
                ]
            ]
            aggregate_trainedRF_max_expert_segments[rule][num_pairs] = [
                max(generation)
                for generation in aggregate_trained_agent_expert_segments[rule][
                    num_pairs
                ]
            ]

    graph_normalized_segments_over_generations(
        best_trueRF_average_expert_segments,
        aggregate_trainedRF_average_expert_segments,
        best_trueRF_max_expert_segments,
        aggregate_trainedRF_max_expert_segments,
    )

    graph_gap_over_pairs(
        best_trueRF_average_expert_segments, aggregate_trainedRF_average_expert_segments
    )


def graph_normalized_segments_over_generations(
    best_trueRF_average_expert_segments,
    aggregate_trainedRF_average_expert_segments,
    best_trueRF_max_expert_segments,
    aggregate_trainedRF_max_expert_segments,
):
    os.makedirs(reward.figure_path, exist_ok=True)

    for rule in best_trueRF_average_expert_segments.keys():
        x_values = range(len(best_trueRF_average_expert_segments[rule]))
        max_avg_trueRF_segments = max(best_trueRF_average_expert_segments[rule])
        max_max_trueRF_segments = max(best_trueRF_max_expert_segments[rule])
        best_trueRF_average_expert_segments[rule] = [
            avg_segments / max_avg_trueRF_segments
            for avg_segments in best_trueRF_average_expert_segments[rule]
        ]
        best_trueRF_max_expert_segments[rule] = [
            max_segments / max_max_trueRF_segments
            for max_segments in best_trueRF_max_expert_segments[rule]
        ]

        for num_pairs in aggregate_trainedRF_average_expert_segments[rule].keys():
            aggregate_trainedRF_average_expert_segments[rule][num_pairs] = [
                avg_segments / max_avg_trueRF_segments
                for avg_segments in aggregate_trainedRF_average_expert_segments[rule][
                    num_pairs
                ]
            ]
        for num_pairs in aggregate_trainedRF_max_expert_segments[rule].keys():
            aggregate_trainedRF_max_expert_segments[rule][num_pairs] = [
                max_segments / max_max_trueRF_segments
                for max_segments in aggregate_trainedRF_max_expert_segments[rule][
                    num_pairs
                ]
            ]

        plt.figure()
        plt.plot(
            x_values, best_trueRF_average_expert_segments[rule], label="Ground Truth"
        )
        for num_pairs in sorted(
            aggregate_trainedRF_average_expert_segments[rule].keys()
        ):
            # if num_pairs == 1000000:
            #     sum = 0
            #     for i in range(len(x_values)):
            #         print(i, best_trueRF_average_expert_segments[rule][i], "\t", aggregate_trainedRF_average_expert_segments[rule][num_pairs][i], "\t", best_trueRF_average_expert_segments[rule][i] - aggregate_trainedRF_average_expert_segments[rule][num_pairs][i], "\t", sum)
            #         sum += best_trueRF_average_expert_segments[rule][i] - aggregate_trainedRF_average_expert_segments[rule][num_pairs][i]
            plt.plot(
                x_values,
                aggregate_trainedRF_average_expert_segments[rule][num_pairs],
                label=f"{num_pairs} pairs",
            )
        plt.xlabel("Generation")
        plt.ylabel("Number of Expert Trajectories (Normalized by GT)")
        plt.title("Ground Truth vs Trained Reward: Average Number of Expert Segments")
        plt.legend()
        plt.savefig(f"{reward.figure_path}average_norm_{rule}_rules.png")
        plt.close()

        plt.figure()
        plt.plot(x_values, best_trueRF_max_expert_segments[rule], label="Ground Truth")
        for num_pairs in aggregate_trainedRF_max_expert_segments[rule].keys():
            plt.plot(
                x_values,
                aggregate_trainedRF_max_expert_segments[rule][num_pairs],
                label=f"{num_pairs} pairs",
            )
        plt.xlabel("Generation")
        plt.ylabel("Number of Expert Trajectories (Normalized by GT)")
        plt.title("Ground Truth vs Trained Reward: Max Number of Expert Segments")
        plt.legend()
        plt.savefig(f"{reward.figure_path}max_norm_{rule}_rules.png")
        plt.close()


def graph_gap_over_pairs(
    best_trueRF_average_expert_segments, aggregate_trainedRF_average_expert_segments
):
    os.makedirs(reward.figure_path, exist_ok=True)
    plt.figure()
    for rule in best_trueRF_average_expert_segments.keys():
        x_values = sorted(aggregate_trainedRF_average_expert_segments[rule].keys())
        y_values = []

        max_avg_trueRF_segments = max(best_trueRF_average_expert_segments[rule])
        best_trueRF_average_expert_segments[rule] = [
            avg_segments / max_avg_trueRF_segments
            for avg_segments in best_trueRF_average_expert_segments[rule]
        ]
        for num_pairs in aggregate_trainedRF_average_expert_segments[rule].keys():
            aggregate_trainedRF_average_expert_segments[rule][num_pairs] = [
                avg_segments / max_avg_trueRF_segments
                for avg_segments in aggregate_trainedRF_average_expert_segments[rule][
                    num_pairs
                ]
            ]

        for num_pairs in x_values:
            if len(best_trueRF_average_expert_segments[rule]) != len(
                aggregate_trainedRF_average_expert_segments[rule][num_pairs]
            ):
                print("Mismatch in generations between GT and Trained!")
            gap = sum(
                [
                    best_trueRF_average_expert_segments[rule][generation]
                    - aggregate_trainedRF_average_expert_segments[rule][num_pairs][
                        generation
                    ]
                    for generation in range(
                        len(best_trueRF_average_expert_segments[rule])
                    )
                ]
            )
            y_values.append(gap / len(best_trueRF_average_expert_segments[rule]))
        plt.plot(x_values, y_values, label=f"{rule} rules")
    plt.xlabel("Number of Trajectory Pairs (Log Scale)")
    plt.xscale("log")
    plt.ylabel("Average Gap in Reward (Reward_GT - Reward_Trained)")
    plt.title("Reward Gap vs. Trajectory Pairs")
    plt.legend()
    plt.savefig(f"{reward.figure_path}gap.png")
    plt.close()

if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Performance plots"
    )
    parse.add_argument(
        "-c",
        "--composition",
        type=int,
        help="number of rules",
    )
    args = parse.parse_args()
    num_rules = args.composition
    best_true_agent_expert_segments, aggregate_trained_agent_expert_segments = unzipper_chungus_deluxe(num_rules)

    handle_plotting_sana(
        best_true_agent_expert_segments,
        aggregate_trained_agent_expert_segments,
    )
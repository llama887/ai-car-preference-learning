import argparse
import glob
import os
import pickle
import re
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import reward

zips_path = "zips/"

T_VALUE_95 = stats.t.ppf((1 + 0.95) / 2, df=19)

from agent import AGENTS_PER_GENERATION


def extract_trajectories(zip_file):
    num_pairs = int(re.search(r"trajectories_t(\d+)*", zip_file).group(1))
    trained_satisfaction_segments = []

    os.makedirs("temp_trajectories", exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall("temp_trajectories")

    trainedRF_files = glob.glob("temp_trajectories/trajectories*/trainedRF_*.pkl")
    if not trainedRF_files:
        raise Exception("TrainedRF file missing")

    with open(trainedRF_files[0], "rb") as f:
        trained_trajectories = pickle.load(f)

    num_trajectories = len(trained_trajectories)

    # Process trained agent trajectories
    count = 0
    while count < len(trained_trajectories):
        gen_trained_satisfaction_segments = [
            trained_trajectories[count + i].num_satisfaction_segments
            for i in range(AGENTS_PER_GENERATION)
        ]
        trained_satisfaction_segments.append(gen_trained_satisfaction_segments)
        count += AGENTS_PER_GENERATION

    trained_satisfaction_segments = np.array(trained_satisfaction_segments)
    return num_pairs, num_trajectories, trained_satisfaction_segments


def unzipper_chungus_deluxe(num_rules, ensembling):
    true_satisfaction_segments = {}
    aggregate_trained_satisfaction_segments = {}
    aggregate_ensemble_trained_satisfaction_segments = {}

    for rule_count in range(1, num_rules + 1):
        rule_aggregate_segments = {}
        rule_aggregate_ensembling_segments = {}

        zip_files = glob.glob(f"{zips_path}trajectories_t*_r{rule_count}.zip")
        print(f"{rule_count} rule files:", zip_files)

        if not zip_files:
            raise Exception("Zip files missing")

        num_trajectories = 0
        for zip_file in zip_files:
            num_pairs, num_trajectories, trained_satisfaction_segments = (
                extract_trajectories(zip_file)
            )
            rule_aggregate_segments[num_pairs] = trained_satisfaction_segments

            shutil.rmtree("temp_trajectories")

        if ensembling:
            ensembling_zip_files = glob.glob(
                f"{zips_path}trajectories_t*_r{rule_count}_ensembling.zip"
            )
            for zip_file in ensembling_zip_files:
                num_pairs, num_trajectories, trained_satisfaction_segments = (
                    extract_trajectories(zip_file)
                )
                rule_aggregate_ensembling_segments[num_pairs] = (
                    trained_satisfaction_segments
                )

        trueRF_path = f"trueRF_trajectories/trueRF_{num_trajectories}_trajectories_{rule_count}_rules.pkl"
        if not os.path.exists(trueRF_path):
            raise Exception(f"TrueRF file not found: {trueRF_path}")
        with open(trueRF_path, "rb") as f:
            true_trajectories = pickle.load(f)
        print(trueRF_path)

        this_rule_true_satisfaction_segments = []
        # Process true agent trajectories
        count = 0
        generations = num_trajectories // AGENTS_PER_GENERATION
        for _ in range(generations):
            gen_true_satisfaction_segments = [
                true_trajectories[count + i].num_satisfaction_segments
                for i in range(AGENTS_PER_GENERATION)
            ]
            this_rule_true_satisfaction_segments.append(gen_true_satisfaction_segments)
            count += AGENTS_PER_GENERATION

        true_satisfaction_segments[rule_count] = np.array(this_rule_true_satisfaction_segments)
        aggregate_trained_satisfaction_segments[rule_count] = rule_aggregate_segments
        aggregate_ensemble_trained_satisfaction_segments[rule_count] = rule_aggregate_ensembling_segments

    # best_true_satisfaction_segments key: rules -> value: best performing trueRF (100 x 20) 100 generations of (# of satisfaction segments by 20 agents))
    # aggregate_trained_satisfaction_segments  key: rules -> value: Map[key: # trajectory pairs -> value: (100 x 20)]
    return (
        true_satisfaction_segments,
        aggregate_trained_satisfaction_segments,
        aggregate_ensemble_trained_satisfaction_segments,
    )


def get_true_generation_averages_and_best_generation(true_satisfaction_segments):
    trueRF_average_satisfaction_segments = {}
    trueRF_best_generation = {}

    for rule, generations in true_satisfaction_segments.items():
        avg_satisfaction = np.mean(generations, axis=1)
        trueRF_average_satisfaction_segments[rule] = avg_satisfaction

        best_gen_index = np.argmax(avg_satisfaction)
        trueRF_best_generation[rule] = generations[best_gen_index]

    return trueRF_average_satisfaction_segments, trueRF_best_generation


def get_trained_generation_averages_and_best_generation(
    aggregate_trained_satisfaction_segments,
):
    aggregate_trainedRF_average_satisfaction_segments = {}
    aggregate_trainedRF_best_generation = {}

    for rule, pairs_dict in aggregate_trained_satisfaction_segments.items():
        aggregate_trainedRF_average_satisfaction_segments[rule] = {}
        aggregate_trainedRF_best_generation[rule] = {}

        for num_pairs, generations in pairs_dict.items():
            avg_satisfaction = np.mean(generations, axis=1)
            aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs] = (
                avg_satisfaction
            )
            best_gen_index = np.argmax(avg_satisfaction)
            aggregate_trainedRF_best_generation[rule][num_pairs] = generations[
                best_gen_index
            ]

    return (
        aggregate_trainedRF_average_satisfaction_segments,
        aggregate_trainedRF_best_generation,
    )


def handle_plotting_sana(
    true_satisfaction_segments,
    aggregate_trained_satisfaction_segments,
    aggregate_ensembling_trained_satisfaction_segments=None,
):
    trueRF_average_satisfaction_segments, trueRF_best_generation = get_true_generation_averages_and_best_generation(true_satisfaction_segments)
    for rules in trueRF_average_satisfaction_segments:
        print(f"{rules} Rules:", trueRF_average_satisfaction_segments[rules].tolist())
    aggregate_trainedRF_average_satisfaction_segments, aggregate_trainedRF_best_generation = get_trained_generation_averages_and_best_generation(aggregate_trained_satisfaction_segments)

    graph_normalized_segments_over_generations(
        trueRF_average_satisfaction_segments,
        aggregate_trainedRF_average_satisfaction_segments,
    )

    graph_gap_over_pairs(trueRF_best_generation, aggregate_trainedRF_best_generation)

    if aggregate_ensembling_trained_satisfaction_segments:
        (
            aggregate_ensembling_trainedRF_average_satisfaction_segments,
            aggregate_ensembling_trainedRF_best_generation,
        ) = get_trained_generation_averages_and_best_generation(
            aggregate_ensembling_trained_satisfaction_segments
        )

        graph_normalized_segments_over_generations(
            trueRF_average_satisfaction_segments,
            aggregate_ensembling_trainedRF_average_satisfaction_segments,
            plot_type="ensembling",
        )

        graph_gap_over_pairs_w_ensembling(
            trueRF_best_generation,
            aggregate_trainedRF_best_generation,
            aggregate_ensembling_trainedRF_best_generation,
        )


def graph_normalized_segments_over_generations(
    trueRF_average_satisfaction_segments,
    aggregate_trainedRF_average_satisfaction_segments,
    plot_type=None,
):
    os.makedirs(reward.figure_path, exist_ok=True)

    for rule in trueRF_average_satisfaction_segments.keys():
        x_values = range(len(trueRF_average_satisfaction_segments[rule]))
        max_avg_trueRF_segments = max(trueRF_average_satisfaction_segments[rule])
        trueRF_average_satisfaction_segments[rule] = [
            avg_segments / max_avg_trueRF_segments
            for avg_segments in trueRF_average_satisfaction_segments[rule]
        ]

        for num_pairs in aggregate_trainedRF_average_satisfaction_segments[rule].keys():
            aggregate_trainedRF_average_satisfaction_segments[rule][num_pairs] = [
                avg_segments / max_avg_trueRF_segments
                for avg_segments in aggregate_trainedRF_average_satisfaction_segments[
                    rule
                ][num_pairs]
            ]

        plt.figure()
        plt.plot(
            x_values, trueRF_average_satisfaction_segments[rule], label="Ground Truth"
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

        figure_title = f"{reward.figure_path}average_norm_{rule}_rules"
        if plot_type:
            figure_title += f"_{plot_type}"
        plt.savefig(f"{figure_title}.png")
        plt.close()


def graph_gap_over_pairs(trueRF_best_generation, aggregate_trainedRF_best_generation):
    aggregate_values = {}

    for rule in trueRF_best_generation.keys():
        aggregate_values[rule] = {}
        x_values = sorted(aggregate_trainedRF_best_generation[rule].keys())
        y_values = []
        yerrs = []

        true_best_avg = np.mean(trueRF_best_generation[rule])
        for num_pairs in x_values:
            trained_best_avg = np.mean(
                aggregate_trainedRF_best_generation[rule][num_pairs]
            )
            gap = (true_best_avg - trained_best_avg) / true_best_avg
            sem_gt = np.std(trueRF_best_generation[rule], ddof=1) / np.sqrt(20)
            sem_trained = np.std(
                aggregate_trainedRF_best_generation[rule][num_pairs], ddof=1
            ) / np.sqrt(20)
            gap_error = (1 / true_best_avg) * np.sqrt(
                sem_gt**2 + (trained_best_avg / true_best_avg) ** 2 * sem_trained**2
            )
            yerr = gap_error * T_VALUE_95
            y_values.append(gap)
            yerrs.append(yerr)

        aggregate_values[rule]["x"] = x_values.copy()
        aggregate_values[rule]["y"] = y_values.copy()
        aggregate_values[rule]["yerrs"] = yerrs.copy()

    os.makedirs(reward.figure_path, exist_ok=True)
    plt.figure()
    for rule in aggregate_values:
        x_values = aggregate_values[rule]["x"]
        y_values = aggregate_values[rule]["y"]
        plt.plot(x_values, y_values, "-o", label=f"{rule} rules")
    plt.xlabel("Number of Trajectory Pairs (Log Scale)")
    plt.xscale("log")
    plt.ylabel("Average Gap in Reward (Reward_GT - Reward_Trained)")
    plt.legend()
    plt.savefig(f"{reward.figure_path}gap.png")
    plt.close()

    os.makedirs(reward.figure_path, exist_ok=True)
    plt.figure()
    for rule in aggregate_values:
        x_values = aggregate_values[rule]["x"]
        y_values = aggregate_values[rule]["y"]
        yerrs = aggregate_values[rule]["yerrs"]
        (line,) = plt.plot(x_values, y_values, "-o", label=f"{rule} rules")
        plt.errorbar(
            x_values,
            y_values,
            yerr=yerrs,
            fmt="o",
            capsize=5,
            alpha=0.3,
            color=line.get_color(),
        )
    plt.xlabel("Number of Trajectory Pairs (Log Scale)")
    plt.xscale("log")
    plt.ylabel("Average Gap in Reward (Reward_GT - Reward_Trained)")
    plt.legend()
    plt.savefig(f"{reward.figure_path}gap_w_error.png")
    plt.close()


def graph_gap_over_pairs_w_ensembling(
    trueRF_best_generation,
    aggregate_trainedRF_best_generation,
    aggregate_ensembling_trainedRF_best_generation,
):
    aggregate_values = {}

    for rule in trueRF_best_generation.keys():
        aggregate_values[rule] = {}
        x_values = sorted(aggregate_trainedRF_best_generation[rule].keys())
        y_values = []

        true_best_avg = np.mean(trueRF_best_generation[rule])
        for num_pairs in x_values:
            trained_best_avg = np.mean(
                aggregate_trainedRF_best_generation[rule][num_pairs]
            )
            ensembling_trained_best_avg = np.mean(
                aggregate_ensembling_trainedRF_best_generation[rule][num_pairs]
            )
            gap = (trained_best_avg - ensembling_trained_best_avg) / true_best_avg
            y_values.append(gap)

        aggregate_values[rule]["x"] = x_values.copy()
        aggregate_values[rule]["y"] = y_values.copy()

    os.makedirs(reward.figure_path, exist_ok=True)
    plt.figure()
    for rule in aggregate_values:
        x_values = aggregate_values[rule]["x"]
        y_values = aggregate_values[rule]["y"]
        plt.plot(x_values, y_values, "-o", label=f"{rule} rules")
    plt.xlabel("Number of Trajectory Pairs (Log Scale)")
    plt.xscale("log")
    plt.ylabel("Average Gap in Reward (Baseline - Ensembling)")
    plt.legend()
    plt.savefig(f"{reward.figure_path}gap_ensembling.png")
    plt.close()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Performance plots")
    parse.add_argument(
        "-c",
        "--composition",
        type=int,
        help="number of rules",
    )
    parse.add_argument(
        "-e",
        "--ensembling",
        action="store_true",
        help="number of rules",
    )
    args = parse.parse_args()
    num_rules = args.composition
    (
        true_satisfaction_segments,
        aggregate_trained_satisfaction_segments,
        aggregate_ensembling_trained_satisfaction_segments,
    ) = unzipper_chungus_deluxe(num_rules, args.ensembling)

    if args.ensembling:
        handle_plotting_sana(
            true_satisfaction_segments,
            aggregate_trained_satisfaction_segments,
            aggregate_ensembling_trained_satisfaction_segments,
        )
    else:
        handle_plotting_sana(
            true_satisfaction_segments,
            aggregate_trained_satisfaction_segments,
        )

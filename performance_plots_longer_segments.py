import argparse
import glob
import os
import pickle
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import reward

zips_path = "zips_test/"
T_VALUE_95 = stats.t.ppf((1 + 0.95) / 2, df=19)
TRAJECTORIES = 1000000
RULES = 3

from agent import AGENTS_PER_GENERATION


def extract_trajectories(zip_file):
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
    return num_trajectories, trained_satisfaction_segments


def unzipper_chungus_deluxe(max_segment_length):
    aggregate_trained_satisfaction_segments = {}

    for segment_length in range(1, max_segment_length + 1):
        zip_file = glob.glob(f"{zips_path}trajectories_s{segment_length}_r{RULES}.zip")[
            0
        ]
        print(f"{segment_length} length:", zip_file)

        if not zip_file:
            raise Exception("Zip files missing")

        num_trajectories, trained_satisfaction_segments = extract_trajectories(zip_file)
        aggregate_trained_satisfaction_segments[segment_length] = (
            trained_satisfaction_segments
        )

        shutil.rmtree("temp_trajectories")
    # best_true_satisfaction_segments key: rules -> value: best performing trueRF (100 x 20) 100 generations of (# of satisfaction segments by 20 agents))
    # aggregate_trained_satisfaction_segments  key: rules -> value: Map[key: # trajectory pairs -> value: (100 x 20)]
    return (aggregate_trained_satisfaction_segments,)


def get_trained_generation_averages_and_best_generation(
    aggregate_trained_satisfaction_segments,
):
    aggregate_trainedRF_average_satisfaction_segments = np.array([])
    aggregate_trainedRF_best_generation = np.array([])

    for (
        segment_length,
        generation_segments,
    ) in aggregate_trained_satisfaction_segments.items():
        aggregate_trainedRF_average_satisfaction_segments[segment_length] = np.mean(
            generation_segments, axis=1
        )
        best_gen_index = np.argmax(
            aggregate_trainedRF_average_satisfaction_segments[segment_length]
        )
        aggregate_trainedRF_best_generation[segment_length] = (
            aggregate_trainedRF_average_satisfaction_segments[segment_length][
                best_gen_index
            ]
        )

    return (
        aggregate_trainedRF_average_satisfaction_segments,
        aggregate_trainedRF_best_generation,
    )


def handle_plotting_sana(
    aggregate_trained_satisfaction_segments,
):
    aggregate_trainedRF_average_satisfaction_segments = {}
    for (
        segment_length,
        generation_segments,
    ) in aggregate_trained_satisfaction_segments.items():
        aggregate_trainedRF_average_satisfaction_segments[segment_length] = np.mean(
            generation_segments, axis=1
        )

    graph_normalized_segments_over_generations(
        aggregate_trainedRF_average_satisfaction_segments
    )
    graph_gap_over_pairs(aggregate_trainedRF_average_satisfaction_segments)


def graph_normalized_segments_over_generations(
    aggregate_trainedRF_average_satisfaction_segments,
):
    os.makedirs(reward.figure_path, exist_ok=True)
    x_values = range(len(aggregate_trainedRF_average_satisfaction_segments[1]))
    for segment_length in sorted(
        aggregate_trainedRF_average_satisfaction_segments.keys()
    ):
        plt.plot(
            x_values,
            aggregate_trainedRF_average_satisfaction_segments[segment_length],
            label=f"{segment_length} length",
        )
    plt.xlabel("Generation")
    plt.ylabel("Ground Truth Reward (wrt GT Agent)")
    plt.legend()

    figure_title = f"{reward.figure_path}average_norm"
    plt.savefig(f"{figure_title}.png")
    plt.close()


def graph_gap_over_pairs(aggregate_trainedRF_average_satisfaction_segments):
    x_values = sorted(aggregate_trainedRF_average_satisfaction_segments.keys())
    y_values = []

    for segment_length in x_values:
        trained_best_avg = np.max(
            aggregate_trainedRF_average_satisfaction_segments[segment_length]
        )
        y_values.append(trained_best_avg)

    os.makedirs(reward.figure_path, exist_ok=True)
    plt.figure()
    plt.plot(x_values, y_values, "-o")
    plt.xlabel("Segment Length")
    plt.ylabel("Average Gap in Reward (Reward_GT - Reward_Trained)")
    plt.legend()
    plt.savefig(f"{reward.figure_path}gap.png")
    plt.close()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Performance plots")
    parse.add_argument(
        "-s",
        "--max_segment_length",
        type=int,
        help="number of rules",
    )
    args = parse.parse_args()
    (aggregate_trained_satisfaction_segments,) = unzipper_chungus_deluxe(
        args.max_segment_length
    )

    handle_plotting_sana(
        aggregate_trained_satisfaction_segments,
    )

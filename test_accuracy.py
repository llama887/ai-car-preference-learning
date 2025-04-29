import torch
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from tqdm import tqdm
import math

import agent
from agent import (
    STATE_ACTION_SIZE
)

import rules

import reward
from reward import (
    Ensemble,
    TrajectoryDataset,
    TrajectoryRewardNet
)

TESTSET_SIZE = 1000000
TEST_DATA_PATH = f'database_gargantuar_testing_1_length.pkl'

def check_dataset(test_file):
    with open(test_file, "rb") as f:
        test_data = pickle.load(f)
        print(len(test_data))
    

    wrong_counter = 0
    for i in range(len(test_data)):
        traj1, traj2, true_pref, score1, score2 = test_data[i]
        if i < 5:
            print(f"Trajectory {i}:", traj1, "Score:", score1)
        rescore1 = rules.check_rules_one(traj1, rules.NUMBER_OF_RULES)[0]
        rescore2 = rules.check_rules_one(traj2, rules.NUMBER_OF_RULES)[0]
        if (rescore1 == rules.NUMBER_OF_RULES) != score1:
            wrong_counter += 1
        if (rescore2 == rules.NUMBER_OF_RULES) != score2:
            wrong_counter += 1
    print(f"Total mismatches before Dataset: {wrong_counter} / {len(test_data)}")

    wrong_counter = 0
    wrong_trajectories = []
    test_dataset = TrajectoryDataset(file_path=test_file, variance_pairs=None, preload=True)
    test_size = len(test_dataset)
    print("TEST SIZE:", test_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = 1,
        shuffle=False,
        pin_memory=False,
        num_workers=4,
    )
    for i in range(len(test_dataloader.dataset)):
        traj1, traj2, true_pref, score1, score2 = test_dataloader.dataset[i]
        # if i < 5:
        #     print(f"Trajectory {i}:", traj1, "Score:", score1)
        rescore1 = rules.check_batch_rules(traj1, rules.NUMBER_OF_RULES, i < 5)[0]
        rescore2 = rules.check_batch_rules(traj2, rules.NUMBER_OF_RULES, False)[0]
        if (rescore1 == rules.NUMBER_OF_RULES) != score1:
            wrong_trajectories.append((test_data[i][0], traj1))
            wrong_counter += 1
        if (rescore2 == rules.NUMBER_OF_RULES) != score2:
            wrong_trajectories.append((test_data[i][1], traj2))
            wrong_counter += 1
    print(f"Total mismatches after Dataset: {wrong_counter} / {len(test_dataloader.dataset)}")
    print("WRONG TRAJECTORIES:")
    for i in range(min(5, len(wrong_trajectories))):
        print(f"OLD {i}:", wrong_trajectories[i][0])
        print("DISTANCE:", math.sqrt((wrong_trajectories[i][0][0].position[0] - wrong_trajectories[i][0][1].position[0]) ** 2 + (wrong_trajectories[i][0][0].position[1] - wrong_trajectories[i][0][1].position[1]) ** 2))
        print(f"NEW {i}:", wrong_trajectories[i][1])
        traj = wrong_trajectories[i][1][0]
        print("DISTANCE:", math.sqrt((traj[6] - traj[14]) ** 2 + (traj[7] - traj[15]) ** 2))
        print()


def generate_testset(test_file):
    agent.paired_database = test_file

    print("TESTSET NEEDS THE FOLLOWING SEGMENTS:")
    agent.display_requested_segments(TESTSET_SIZE)
    saved_segments = agent.load_from_garg(TEST_DATA_PATH)
    if not agent.finished_collecting(saved_segments, TESTSET_SIZE):
        raise Exception("Not enough segments to generate test set.")

    print(f"Test set sampling complete. Generating test set with {TESTSET_SIZE} trajectory pairs...")
    agent.generate_database(agent.trajectories_path, TESTSET_SIZE, saved_segments, "segments", segment_generation_mode="random")
    

def load_models(reward_paths, hidden_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(reward_paths) == 1:
        print("\nLoading reward network...")
        reward_network = TrajectoryRewardNet(
            STATE_ACTION_SIZE * (agent.train_trajectory_length + 1),
            hidden_size=hidden_size,
        ).to(device)
        weights = torch.load(reward_paths[0], map_location=torch.device(f"{device}"))
        reward_network.load_state_dict(weights)
        return reward_network
    else:
        if reward_paths[0] == "QUICK":
            if len(reward_paths) > 2:
                raise Exception("REWARD PATH ERROR (QUICK MODE)")
            ensemble_dir = reward_paths[1] + "*"
            reward_paths = []
            for file in glob.glob(ensemble_dir):
                reward_paths.append(file)

        print(f"\nLoading ensemble of {len(reward_paths)} models...")
        ensemble_nets = [
            TrajectoryRewardNet(
                STATE_ACTION_SIZE * (agent.train_trajectory_length + 1),
                hidden_size=hidden_size,
            ).to(device)
            for _ in range(len(reward_paths))
        ]
        ensemble_weights = []
        for reward_path in reward_paths:
            ensemble_weights.append(
                torch.load(reward_path, map_location=torch.device(f"{device}"))
            )
        for i in range(len(ensemble_nets)):
            ensemble_nets[i].load_state_dict(ensemble_weights[i])
            print(f"Loaded model #{i} from ensemble...")
        ensemble = Ensemble(
            STATE_ACTION_SIZE * (agent.train_trajectory_length + 1),
            len(ensemble_nets),
            ensemble_nets,
        )
        return ensemble

def test_model(model_path, hidden_size, batch_size=256):
    num_rules = rules.NUMBER_OF_RULES

    if not model_path:
        raise Exception("Model not found...")
    
    model = load_models(model_path, hidden_size)

    test_file = f"database_test_{num_rules}_rules.pkl"
    generate_testset(test_file)

    test_dataset = TrajectoryDataset(file_path=test_file, variance_pairs=None, preload=True)

    test_size = len(test_dataset)
    print("TEST SIZE:", test_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = test_size if test_size < batch_size else batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=4,
    )

    segment_rules_satisfied = [[],[]]
    segment_rewards = [[], []]
    segment_true_scores = [[], []]

    with torch.no_grad():
        for test_traj1, test_traj2, test_true_pref, test_score1, test_score2 in tqdm(test_dataloader, desc="Testing Model"):
            test_rewards1 = model(test_traj1)
            test_rewards2 = model(test_traj2)

            segment_rules_satisfied[0].extend(rules.check_batch_rules(test_traj1, num_rules))
            segment_rules_satisfied[1].extend(rules.check_batch_rules(test_traj2, num_rules))
            segment_rewards[0].extend(test_rewards1.tolist())
            segment_rewards[1].extend(test_rewards2.tolist())
            segment_true_scores[0].extend(test_score1.tolist())
            segment_true_scores[1].extend(test_score2.tolist())

    n_pairs = len(segment_rewards[0])
    for i in range(n_pairs):
        segment_rewards[0][i] = segment_rewards[0][i][0][0]
        segment_rewards[1][i] = segment_rewards[1][i][0][0]


    print("\nVerifying true scores match rules satisfied:")
    mismatches = 0
    for idx in range(n_pairs):
        for traj_idx in [0, 1]:
            expected_score = 1 if segment_rules_satisfied[traj_idx][idx] == num_rules else 0
            actual_score = segment_true_scores[traj_idx][idx]
            if expected_score != actual_score:
                mismatches += 1
                print(f"rule issue: Trajectory {idx}-{traj_idx} has {segment_rules_satisfied[traj_idx][idx]} rules satisfied but score={actual_score}")

    if mismatches == 0:
        print("All scores correctly match the number of rules satisfied!")
    else:
        print(f"Found {mismatches} mismatches between rules satisfied and true scores.")

    # segment_rules_satisfied_types = set()
    # for elem in segment_rules_satisfied:
    #     segment_rules_satisfied_types.add(type(elem))
    # print(segment_rules_satisfied_types)

    # segment_rewards_types = set()
    # for elem in segment_rewards[0]:
    #     segment_rewards_types.add(type(elem))
    # print(segment_rewards_types)

    total_correct = 0
    total_diff = 0
    total_adjusted_correct = 0
    acc_pairings = [[0, 0] * num_rules for _ in range(num_rules)]
    error = 0

    for i in range(n_pairs):
        different_reward = (segment_true_scores[0][i] != segment_true_scores[1][i])
        if different_reward and segment_rules_satisfied[0][i] != num_rules and segment_rules_satisfied[1][i] != num_rules:
            # print(segment_rules_satisfied[0][i], segment_rules_satisfied[1][i])
            error += 1
        prediction = segment_rewards[0][i] >= segment_rewards[1][i]
        true_pref = segment_true_scores[0][i] >= segment_true_scores[1][i]
        correct = prediction == true_pref
        adjusted_correct = different_reward * correct

        if segment_rules_satisfied[0][i] == num_rules and segment_rules_satisfied[1][i] != num_rules:
            acc_pairings[segment_rules_satisfied[1][i]][0] += adjusted_correct
            acc_pairings[segment_rules_satisfied[1][i]][1] += 1
        if segment_rules_satisfied[1][i] == num_rules and segment_rules_satisfied[0][i] != num_rules:
            acc_pairings[segment_rules_satisfied[0][i]][0] += adjusted_correct
            acc_pairings[segment_rules_satisfied[0][i]][1] += 1
        
        total_correct += correct
        total_diff += different_reward
        total_adjusted_correct += adjusted_correct

    test_acc = total_correct / n_pairs
    print(total_adjusted_correct, "/", total_diff)
    print(error)
    adjusted_test_acc = total_adjusted_correct / total_diff if total_diff > 0 else 0

    segment_rules_satisfied = segment_rules_satisfied[0] + segment_rules_satisfied[1]
    segment_rewards = segment_rewards[0] + segment_rewards[1]

    df = pd.DataFrame(
        {
            "Rules Satisfied": segment_rules_satisfied,
            "Reward of Trajectory Segment": segment_rewards,
        }
    )

    sns.violinplot(
        x="Rules Satisfied",
        y="Reward of Trajectory Segment",
        data=df,
        inner="box",
        palette="muted",
        alpha=0.55,
    )

    title = "test_violin"
    plt.legend()
    plt.savefig(f"{reward.figure_path}{title}.png", dpi=600)
    plt.close()

    with open(f"{agent.trajectories_path}/violin_data.pkl", "wb") as f:
        pickle.dump(df, f)


    print("TEST ACCURACY:", test_acc, "ADJUSTED TEST ACCURACY:", adjusted_test_acc)
    print("ACCURACY BREAKDOWN:")
    for i in range(len(acc_pairings)):
        print(f"{i} RULES vs. {num_rules} RULES (SATISFACTION):[{acc_pairings[i][0]} / {acc_pairings[i][1]}] ({acc_pairings[i][0] / acc_pairings[i][1]})")
    return test_acc, adjusted_test_acc, acc_pairings

# rules.NUMBER_OF_RULES = 1
# rules.RULES_INCLUDED = [2]
# rules.SEGMENT_DISTRIBUTION_BY_RULES = [1/2, 1/2]
# test_model(
#     model_path=["models/model_3000_epochs_1000000_pairs_1_rules.pth"],
#     hidden_size=952,
#     batch_size=6032)

    
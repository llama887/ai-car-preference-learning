import torch
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from tqdm import tqdm

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

def test_model(model_path, test_file, hidden_size, batch_size=256):
    if not model_path:
        raise Exception("Model not found...")
    if not test_file:
        raise Exception("Test file not found...")
    
    model = load_models(model_path, hidden_size)

    total_correct = 0
    total_diff = 0
    adjusted_correct = 0

    test_dataset = TrajectoryDataset(file_path=test_file, variance_pairs=None, preload=True)
    test_size = len(test_dataset)
    print("TEST SIZE:", test_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = test_size if test_size < batch_size else batch_size,
        shuffle=False,
        pin_memory=False,
    )

    segment_rules_satisfied = []
    segment_rewards = []
    with torch.no_grad():
        for test_traj1, test_traj2, test_true_pref, test_score1, test_score2 in tqdm(test_dataloader, desc="Testing Model:"):
            test_rewards1 = model(test_traj1)
            test_rewards2 = model(test_traj2)

            segment_rules_satisfied.extend(rules.check_batch_rules(test_traj1, rules.NUMBER_OF_RULES))
            segment_rules_satisfied.extend(rules.check_batch_rules(test_traj2, rules.NUMBER_OF_RULES))
            segment_rewards.extend(test_rewards1.tolist())
            segment_rewards.extend(test_rewards2.tolist())

            predictions = (test_rewards1 >= test_rewards2).squeeze()
            correct_predictions = (predictions == test_true_pref).sum().item()
            total_correct += correct_predictions

            different_rewards = (test_score1 != test_score2)
            num_diff_in_batch = different_rewards.sum().item()
            total_diff += num_diff_in_batch

            adjusted_correct += (different_rewards & (predictions == test_true_pref)).sum().item()
            
        test_acc = total_correct / test_size
        adjusted_test_acc = adjusted_correct / total_diff if total_diff > 0 else 0



    for i in range(len(segment_rewards)):
        segment_rewards[i] = segment_rewards[i][0][0]

    # segment_rules_satisfied_types = set()
    # for elem in segment_rules_satisfied:
    #     segment_rules_satisfied_types.add(type(elem))
    # print(segment_rules_satisfied_types)

    # segment_rewards_types = set()
    # for elem in segment_rewards:
    #     segment_rewards_types.add(type(elem))
    # print(segment_rewards_types)

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

    return test_acc, adjusted_test_acc

    
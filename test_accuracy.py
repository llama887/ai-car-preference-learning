import torch
import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import agent
from agent import (
    STATE_ACTION_SIZE
)
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

    test_dataset = TrajectoryDataset(test_file)
    test_size = len(test_dataset)
    print("TEST SIZE:", test_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = test_size if test_size < batch_size else batch_size,
        shuffle=False,
        pin_memory=False,
    )

    with torch.no_grad():
        for test_traj1, test_traj2, test_true_pref, test_score1, test_score2 in test_dataloader:
            test_rewards1 = model(test_traj1)
            test_rewards2 = model(test_traj2)
            predictions = (test_rewards1 >= test_rewards2).squeeze()
            correct_predictions = (predictions == test_true_pref).sum().item()
            total_correct += correct_predictions

            different_rewards = (test_score1 != test_score2)
            num_diff_in_batch = different_rewards.sum().item()
            total_diff += num_diff_in_batch

            adjusted_correct += (different_rewards & (predictions == test_true_pref)).sum().item()
            
        test_acc = total_correct / test_size
        adjusted_test_acc = adjusted_correct / total_diff if total_diff > 0 else 0
    return test_acc, adjusted_test_acc
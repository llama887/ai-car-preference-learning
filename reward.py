import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import wandb
from torch.optim.lr_scheduler import StepLR
import torch.optim.lr_scheduler as lr_scheduler
import optuna
import glob
import os
import shutil
import optuna

# plot the distribution of reward distributions for when the net gets it right and wrong

study = ""


class TrajectoryRewardNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_prob=0.5):
        super(TrajectoryRewardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class TrajectoryDataset(Dataset):
    def __init__(self, file_path):
        self.data_path = file_path
        with open(file_path, "rb") as f:
            self.trajectory_pairs = pickle.load(f)

    def __getitem__(self, idx):
        traj1, traj2, preference, score1, score2 = self.trajectory_pairs[idx]
        # Convert trajectories and preference to tensors
        traj1 = prepare_single_trajectory(traj1)
        traj2 = prepare_single_trajectory(traj2)
        preference = torch.tensor(preference, dtype=torch.float32)
        score1 = torch.tensor(score1, dtype=torch.float32)
        score2 = torch.tensor(score2, dtype=torch.float32)

        # Return the trajectories, preference, and scores
        return traj1, traj2, preference, score1, score2

    def __len__(self):
        return len(self.trajectory_pairs)


def bradley_terry_model(r1, r2):
    exp_r1 = torch.exp(r1)
    exp_r2 = torch.exp(r2)
    probability = exp_r1 / (exp_r1 + exp_r2)
    return probability.squeeze()


def preference_loss(predicted_probabilities, true_preferences):
    return F.binary_cross_entropy(predicted_probabilities, true_preferences)


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    triples = [element[:3] for element in data]
    true_rewards = [element[-2:] for element in data]
    return triples, true_rewards


def prepare_data(data, max_length=450):
    def pad_or_truncate(trajectory, length):
        if len(trajectory) > length:
            return trajectory[:length]
        else:
            padding = [trajectory[-1]] * (length - len(trajectory))
            return trajectory + padding

    trajectories1 = []
    trajectories2 = []
    true_preferences = []
    for t1, t2, preference in data:
        t1_padded = pad_or_truncate(t1, max_length)
        t2_padded = pad_or_truncate(t2, max_length)
        t1_flat = [item for sublist in t1_padded for item in sublist]
        t2_flat = [item for sublist in t2_padded for item in sublist]
        trajectories1.append(t1_flat)
        trajectories2.append(t2_flat)
        true_preferences.append(preference)
    trajectories1 = torch.tensor(trajectories1, dtype=torch.float32)
    trajectories2 = torch.tensor(trajectories2, dtype=torch.float32)
    true_preferences = torch.tensor(true_preferences, dtype=torch.float32)
    return trajectories1, trajectories2, true_preferences


def visualize_trajectories(batch_1, batch_2):
    for trajectory in batch_1:
        x = trajectory[::2]
        y = trajectory[1::2]
        plt.plot(x, y, color="red", alpha=0.1)
    for trajectory in batch_2:
        x = trajectory[::2]
        y = trajectory[1::2]
        plt.plot(x, y, color="blue", alpha=0.1)
    plt.show()


def prepare_single_trajectory(trajectory, max_length=450):
    def pad_or_truncate(trajectory, length):
        if len(trajectory) > length:
            return trajectory[:length]
        else:
            padding = [trajectory[-1]] * (length - len(trajectory))
            return trajectory + padding

    # Pad or truncate the trajectory
    trajectory_padded = pad_or_truncate(trajectory, max_length)
    # Flatten the list of tuples
    trajectory_flat = [item for sublist in trajectory_padded for item in sublist]

    # Convert to tensor and add an extra dimension
    trajectory_tensor = torch.tensor([trajectory_flat], dtype=torch.float32)

    return trajectory_tensor


def calculate_accuracy(predicted_probabilities, true_preferences):
    predicted_preferences = (predicted_probabilities > 0.5).float()
    correct_predictions = (predicted_preferences == true_preferences).float().sum()
    accuracy = correct_predictions / true_preferences.size(0)
    return accuracy.item()


def train_model(
    file_path, net, epochs=1000, optimizer=None, batch_size=32, model_path="best.pth"
):
    wandb.init(project="Micro Preference")
    wandb.watch(net, log="all")

    # Create the dataset
    full_dataset = TrajectoryDataset(file_path)

    # Define the split ratio
    train_ratio = 0.8
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    # Initialize Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(
        val_dataset, batch_size=val_dataset_size, shuffle=False
    )

    # Log label balance
    # Alex: after adding dataloaders, logging this is kinda annoying
    # wandb.log(
    #     {
    #         "True Preferences Training": wandb.Histogram(
    #             true_preferences.detach().cpu().numpy()
    #         ),
    #         "True Preferences Validation": wandb.Histogram(
    #             validation_true_preferences.detach().cpu().numpy()
    #         ),
    #     }
    # )

    if batch_size > dataset_size:
        batch_size = dataset_size

    best_loss = np.inf
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs
    )

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_probability = 0.0

        TP_rewards = []
        TN_rewards = []
        FP_rewards = []
        FN_rewards = []

        for (
            batch_traj1,
            batch_traj2,
            batch_true_pref,
            batch_score1,
            batch_score2,
        ) in train_dataloader:
            rewards1 = net(batch_traj1)
            rewards2 = net(batch_traj2)

            predicted_probabilities = bradley_terry_model(rewards1, rewards2)
            total_probability += predicted_probabilities.sum().item()

            loss = preference_loss(predicted_probabilities, batch_true_pref)
            total_loss += loss.item()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(net.state_dict(), model_path)

            accuracy = calculate_accuracy(predicted_probabilities, batch_true_pref)
            total_accuracy += accuracy * batch_true_pref.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            scheduler.step()

            # Classify rewards
            for idx, prob in enumerate(predicted_probabilities):
                if prob > 0.5 and batch_true_pref[idx] == 1:
                    TP_rewards.append(batch_score1[idx])
                elif prob <= 0.5 and batch_true_pref[idx] == 0:
                    TN_rewards.append(batch_score2[idx])
                elif prob > 0.5 and batch_true_pref[idx] == 0:
                    FP_rewards.append(batch_score2[idx])
                elif prob <= 0.5 and batch_true_pref[idx] == 1:
                    FN_rewards.append(batch_score1[idx])

        average_training_loss = total_loss / (dataset_size // batch_size)
        training_losses.append(average_training_loss)

        average_training_accuracy = total_accuracy / dataset_size
        training_accuracies.append(average_training_accuracy)

        net.eval()

        with torch.no_grad():
            for (
                validation_traj1,
                validation_traj2,
                validation_true_pref,
                validation_score1,
                validation_score2,
            ) in validation_dataloader:
                validation_rewards1 = net(validation_traj1)
                validation_rewards2 = net(validation_traj2)
                validation_predicted_probabilities = bradley_terry_model(
                    validation_rewards1, validation_rewards2
                )
                validation_loss = preference_loss(
                    validation_predicted_probabilities, validation_true_pref
                )
                validation_losses.append(validation_loss.item())

                validation_accuracy = calculate_accuracy(
                    validation_predicted_probabilities, validation_true_pref
                )
                validation_accuracies.append(validation_accuracy)

        wandb.log(
            {
                "Train Loss": average_training_loss,
                "Validation Loss": validation_loss.item(),
                "Train Accuracy": average_training_accuracy,
                "Validation Accuracy": validation_accuracy,
            },
            step=epoch,
        )

        # Log weights histogram
        for name, param in net.named_parameters():
            wandb.log(
                {f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy())},
                step=epoch,
            )

        # Log reward distributions
        if TP_rewards:
            wandb.log(
                {"TP Reward Distribution": wandb.Histogram(np.array(TP_rewards))},
                step=epoch,
            )
        if TN_rewards:
            wandb.log(
                {"TN Reward Distribution": wandb.Histogram(np.array(TN_rewards))},
                step=epoch,
            )
        if FP_rewards:
            wandb.log(
                {"FP Reward Distribution": wandb.Histogram(np.array(FP_rewards))},
                step=epoch,
            )
        if FN_rewards:
            wandb.log(
                {"FN Reward Distribution": wandb.Histogram(np.array(FN_rewards))},
                step=epoch,
            )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {average_training_loss}, Val Loss: {validation_loss.item()}, Train Acc: {average_training_accuracy}, Val Acc: {validation_accuracy}"
            )

    plt.figure()
    plt.plot(training_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/loss.png")

    plt.figure()
    plt.plot(training_accuracies, label="Train Accuracy")
    plt.plot(validation_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("figures/accuracy.png")

    return best_loss


def run_study(file_path, epochs):
    global study
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("file_path", file_path)
    study.set_user_attr("epochs", epochs)
    study.optimize(objective, n_trials=1)


def objective(trial):
    input_size = 450 * 2
    hidden_size = trial.suggest_int("hidden_size", 64, 512)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3)

    net = TrajectoryRewardNet(input_size, hidden_size)
    for param in net.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = train_model(
        file_path=study.user_attrs["file_path"],
        net=net,
        epochs=study.user_attrs["epochs"],
        optimizer=optimizer,
    )

    # Save the best model parameters
    if trial.should_prune():
        raise optuna.TrialPruned()
    else:
        # Save model state with trial number to avoid overwrite
        torch.save(net.state_dict(), f"best_model_trial_{trial.number}.pth")

    return best_loss


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )
    parse.add_argument(
        "-d", "--database", type=str, help="Directory to trajectory database file"
    )
    parse.add_argument(
        "-e", "--epochs", type=int, help="Number of epochs to train the model"
    )
    args = parse.parse_args()
    if args.database:
        file_path = args.database
    else:
        file_path = "trajectories/database_350.pkl"

    try:
        shutil.rmtree("./wandb")
    except OSError as e:
        pass
    for file in glob.glob("best_model*.pth"):
        os.remove(file)

    input_size = 450 * 2
    hidden_size = 128
    learning_rate = 0.0003

    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 1000

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    # Load and print the best trial
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Value: {best_trial.value}")
    print(f"Params: {best_trial.params}")

    # Load the best model
    best_model = TrajectoryRewardNet(input_size, best_trial.params["hidden_size"])
    best_model.load_state_dict(torch.load(f"best_model_trial_{best_trial.number}.pth"))
    torch.save(
        best_model.state_dict(), f"best_model_{best_trial.params['hidden_size']}.pth"
    )

    # Delete saved hyperparameter trials
    for file in glob.glob("best_model_trial*.pth"):
        os.remove(file)

    os.remove("best.pth")

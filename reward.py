import argparse
import glob
import os
import pickle

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split

import wandb

os.environ["WANDB_SILENT"] = "true"
INPUT_SIZE = 2 * 2

figure_path = "figures/"
os.makedirs(figure_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = StandardScaler()


class TrajectoryRewardNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_prob=0.0):
        super(TrajectoryRewardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.ln4 = nn.LayerNorm(hidden_size // 4)
        self.dropout4 = nn.Dropout(dropout_prob)
        self.fc5 = nn.Linear(hidden_size // 4, 1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.relu(self.ln4(self.fc4(x)))
        x = self.dropout4(x)
        x = self.fc5(x)
        # x += 15
        return x


class TrajectoryDataset(Dataset):
    def __init__(self, file_path):
        self.data_path = file_path
        with open(file_path, "rb") as f:
            self.trajectory_pairs = pickle.load(f)
        self.first_trajectories = []
        self.second_trajectories = []
        self.labels = []
        self.score1 = []
        self.score2 = []
        all_data = []
        for trajectory_pair in self.trajectory_pairs:
            trajectory1_flat = [
                [item for sublist in trajectory_pair[0] for item in sublist]
            ]
            trajectory2_flat = [
                [item for sublist in trajectory_pair[1] for item in sublist]
            ]
            temp_flat = [item for sublist in trajectory_pair[0] for item in sublist]
            all_data.append(temp_flat)
            temp_flat = [item for sublist in trajectory_pair[1] for item in sublist]
            all_data.append(temp_flat)
            self.first_trajectories.append(trajectory1_flat)
            self.second_trajectories.append(trajectory2_flat)
            self.labels.append(trajectory_pair[2])
            self.score1.append(trajectory_pair[3])
            self.score2.append(trajectory_pair[4])
        # scaler.fit(all_data)
        # with open("scaler.pkl", "wb") as f:
        #     pickle.dump(scaler, f)
        self.first_trajectories = torch.tensor(
            self.first_trajectories, dtype=torch.float32
        ).to(device)
        self.second_trajectories = torch.tensor(
            self.second_trajectories, dtype=torch.float32
        ).to(device)
        self.labels = torch.tensor(self.labels, dtype=torch.float32).to(device)
        self.score1 = torch.tensor(self.score1, dtype=torch.float32).to(device)
        self.score2 = torch.tensor(self.score2, dtype=torch.float32).to(device)

    def __getitem__(self, idx):
        traj1 = self.first_trajectories[idx]
        traj2 = self.second_trajectories[idx]
        preference = self.labels[idx]
        score1 = self.score1[idx]
        score2 = self.score2[idx]
        return traj1, traj2, preference, score1, score2

    def __len__(self):
        return len(self.trajectory_pairs)


def bradley_terry_model(r1, r2):
    exp_r1 = torch.exp(r1)
    exp_r2 = torch.exp(r2)
    probability = exp_r2 / (exp_r1 + exp_r2)
    return probability.squeeze()


def preference_loss(predicted_probabilities, true_preferences):
    return F.binary_cross_entropy(predicted_probabilities, true_preferences)


def prepare_single_trajectory(trajectory, max_length=2):
    def truncate(trajectory, max_length):
        if len(trajectory) > max_length:
            return trajectory[-max_length:]
        return trajectory

    trajectory_flat = [
        item for sublist in truncate(trajectory, max_length) for item in sublist
    ]

    # Apply the fitted scaler to the flattened trajectory
    # trajectory_flat_whitened = scaler.transform([trajectory_flat])

    # Convert to tensor and add an extra dimension
    trajectory_tensor = torch.tensor(trajectory_flat, dtype=torch.float32).to(device)

    return trajectory_tensor


def calculate_accuracy(predicted_probabilities, true_preferences):
    predicted_preferences = (predicted_probabilities > 0.5).float()
    correct_predictions = (predicted_preferences == true_preferences).float().sum()
    accuracy = correct_predictions / true_preferences.size(0)
    return accuracy.item()


def train_model(
    file_path,
    net,
    epochs=1000,
    optimizer=None,
    batch_size=256,
    model_path="best.pth",
):
    print("BATCH_SIZE:", batch_size)
    wandb.init(project="Micro Preference")
    wandb.watch(net, log="all")

    # Create the dataset
    full_dataset = TrajectoryDataset(file_path)

    # Define the split ratio
    train_ratio = 0.8
    # val_ratio = 0.2
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Initialize Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_size if train_size < batch_size else batch_size,
        shuffle=True,
        pin_memory=False,
    )
    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=val_size if val_size < batch_size else batch_size,
        shuffle=False,
        pin_memory=False,
    )

    if batch_size > train_size:
        batch_size = train_size

    best_loss = np.inf
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs
    )

    epoch = 0
    try:
        while epoch < epochs:
            net.eval()
            total_validation_loss = 0.0
            total_validation_accuracy = 0.0
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
                    validation_true_pref_dist = bradley_terry_model(
                        validation_score1, validation_score2
                    )
                    total_validation_loss += preference_loss(
                        validation_predicted_probabilities, validation_true_pref_dist
                    )
                    total_validation_accuracy += calculate_accuracy(
                        validation_predicted_probabilities, validation_true_pref
                    ) * validation_true_pref.size(0)

            average_validation_loss = total_validation_loss / val_size
            if average_validation_loss < best_loss:
                best_loss = average_validation_loss
                torch.save(net.state_dict(), model_path)
            average_validation_accuracy = total_validation_accuracy / val_size
            validation_losses.append(average_validation_loss.item())
            validation_accuracies.append(average_validation_accuracy)

            net.train()
            total_loss = 0.0
            total_accuracy = 0.0
            total_probability = 0.0

            for (
                batch_traj1,
                batch_traj2,
                batch_true_pref,
                batch_score1,
                batch_score2,
            ) in train_dataloader:
                # for item in list(zip(batch_traj1, batch_traj2, batch_true_pref)):
                #     print(item)
                rewards1 = net(batch_traj1)
                rewards2 = net(batch_traj2)
                # print(rewards1, rewards2)

                predicted_probabilities = bradley_terry_model(rewards1, rewards2)
                batch_true_pref_dist = bradley_terry_model(batch_score1, batch_score2)
                total_probability += predicted_probabilities.sum().item()


                # try:
                #     assert all(0 <= pref <= 1 for pref in predicted_probabilities)
                # except:
                #     print("PREDICTED bradley terry failed.")

                # try:
                #     assert all(0 <= pref <= 1 for pref in batch_true_pref_dist)
                # except:
                #     print("TRUE PREFS bradley terry failed.")

                loss = preference_loss(predicted_probabilities, batch_true_pref_dist)
                total_loss += loss.item()
                # ipdb.set_trace()

                # if loss.item() < best_loss:
                #     best_loss = loss.item()
                #     torch.save(net.state_dict(), model_path)

                accuracy = calculate_accuracy(predicted_probabilities, batch_true_pref)
                total_accuracy += accuracy * batch_true_pref.size(0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                scheduler.step()

            average_training_loss = total_loss / train_size
            training_losses.append(average_training_loss)

            average_training_accuracy = total_accuracy / train_size
            training_accuracies.append(average_training_accuracy)

            wandb.log(
                {
                    "Train Loss": average_training_loss,
                    "Validation Loss": average_validation_loss.item(),
                    "Train Accuracy": average_training_accuracy,
                    "Validation Accuracy": average_validation_accuracy,
                },
                step=epoch,
            )

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs}, Train Loss: {average_training_loss}, Val Loss: {average_validation_loss.item()}, Train Acc: {average_training_accuracy}, Val Acc: {average_validation_accuracy}"
                )
            epoch += 1
    except:
        torch.save(net.state_dict(), f"model_{epoch}.pth")

    plt.figure()
    plt.plot(training_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/loss.png")
    plt.close()

    plt.figure()
    plt.plot(training_accuracies, label="Train Accuracy")
    plt.plot(validation_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("figures/accuracy.png")
    plt.close()

    return best_loss


def train_reward_function(trajectories_file_path, epochs, parameters_path=None):
    input_size = INPUT_SIZE
    if not parameters_path:
        for f in glob.glob("best_model_*.pth"):
            os.remove(f)
        print("RUNNING WITH OPTUNA:")
        global study
        study = optuna.create_study(direction="minimize")
        study.set_user_attr("file_path", trajectories_file_path)
        study.set_user_attr("epochs", epochs)
        study.optimize(objective, n_trials=10)

        # Load and print the best trial
        best_trial = study.best_trial
        print(f"Best trial: {best_trial.number}")
        print(f"Value: {best_trial.value}")
        print(f"Params: {best_trial.params}")

        with open("best_params.yaml", "w") as f:
            yaml.dump(best_trial.params, f)
            print("Best hyperparameters saved.")

        # Load the best model
        best_model = TrajectoryRewardNet(
            input_size, best_trial.params["hidden_size"]
        ).to(device)
        best_model.load_state_dict(
            torch.load(f"best_model_trial_{best_trial.number}.pth")
        )
        torch.save(
            best_model.state_dict(),
            f"best_model_{best_trial.params['hidden_size']}.pth",
        )

        # Delete saved hyperparameter trials
        for file in glob.glob("best_model_trial_*.pth"):
            os.remove(file)

    else:
        print("RUNNING WITHOUT OPTUNA:")
        with open(parameters_path, "r") as file:
            data = yaml.safe_load(file)
            hidden_size = data["hidden_size"]
            learning_rate = data["learning_rate"]
            weight_decay = data["weight_decay"]
            dropout_prob = data["dropout_prob"]
            batch_size = data["batch_size"]

            net = TrajectoryRewardNet(input_size, hidden_size, dropout_prob).to(device)
            for param in net.parameters():
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))
            optimizer = torch.optim.Adam(
                net.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            best_loss = train_model(
                file_path=trajectories_file_path,
                net=net,
                epochs=epochs,
                optimizer=optimizer,
                batch_size=batch_size,
            )
            torch.save(net.state_dict(), f"model_{epochs}.pth")
        return best_loss


def objective(trial):
    input_size = INPUT_SIZE
    hidden_size = trial.suggest_int("hidden_size", 64, 1024)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3)
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.0)
    batch_size = trial.suggest_int("batch_size", 128, 2048)

    net = TrajectoryRewardNet(input_size, hidden_size, dropout_prob=0).to(device)
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
        batch_size=batch_size,
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
        "-d",
        "--database",
        type=str,
        help="Directory to trajectory database file",
    )
    parse.add_argument(
        "-e", "--epochs", type=int, help="Number of epochs to train the model"
    )
    parse.add_argument(
        "-p",
        "--parameters",
        type=str,
        help="Directory to hyperparameter yaml file",
    )
    args = parse.parse_args()
    if args.database:
        data_path = args.database
    else:
        data_path = "trajectories/database_350.pkl"

    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 1000

    train_reward_function(data_path, epochs, args.parameters)

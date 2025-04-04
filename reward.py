import argparse
import glob
import math
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

import rules
import wandb

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["WANDB_SILENT"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

INPUT_SIZE = 8 * 2
OPTUNA_N_TRIALS = 10

figure_path = "figures/"
models_path = "models/"
ensemble_path = None
os.makedirs(figure_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)


train_ratio = 0.8
val_ratio = 0.2
n_pairs = 0

ENSEMBLE_SIZE = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tensors(tensors):
    return tuple([tensor.to(device) for tensor in tensors])

class TrajectoryRewardNet(nn.Sequential):
    def __init__(self, input_size, hidden_size=128, dropout_prob=0.0):
        super(TrajectoryRewardNet, self).__init__(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 4, 1))

        self.best_loss = np.inf
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []

    # def forward(self, x):
    #     x = F.relu(self.ln1(self.fc1(x)))
    #     x = self.dropout1(x)
    #     x = F.relu(self.ln2(self.fc2(x)))
    #     x = self.dropout2(x)
    #     x = F.relu(self.ln3(self.fc3(x)))
    #     x = self.dropout3(x)
    #     x = F.relu(self.ln4(self.fc4(x)))
    #     x = self.dropout4(x)
    #     x = F.relu(self.ln5(self.fc5(x)))
    #     x = self.dropout5(x)
    #     x = F.relu(self.ln6(self.fc6(x)))
    #     x = self.dropout6(x)
    #     x = self.fc7(x)
    #     return x


class Ensemble(nn.Module):
    def __init__(
        self,
        input_size,
        num_models=3,
        models_list=None,
        hidden_size=128,
        dropout_prob=0.0,
    ):
        super(Ensemble, self).__init__()
        self.num_models = num_models
        if not models_list:
            self.model_list = nn.ModuleList(
                [
                    TrajectoryRewardNet(input_size, hidden_size, dropout_prob)
                    for _ in range(self.num_models)
                ]
            )
        else:
            self.model_list = nn.ModuleList(models_list)

        # self.load_models_to_gpu()

    def forward(self, x):
        ret = []
        for model in self.model_list:
            x = x.to(next(model.parameters()).device)  
            ret.append(model(x))
        return sum(ret) / len(ret)

    def load_models_to_gpu(self):
        for i in range(self.num_models):
            self.model_list[i] = self.model_list[i].to(device)
    
    def unload_models(self):
        for i in range(self.num_models):
            self.model_list[i] = self.model_list[i].to("cpu")
        torch.cuda.empty_cache()


class TrajectoryDataset(Dataset):
    def __init__(self, file_path, variance_pairs=None, preload=True):
        self.first_trajectories = []
        self.second_trajectories = []
        self.labels = []
        self.score1 = []
        self.score2 = []

        self.data_path = file_path
        if variance_pairs:
            self.first_trajectories = variance_pairs["first_trajectories"]
            self.second_trajectories = variance_pairs["second_trajectories"]
            self.labels = variance_pairs["labels"]
            self.score1 = variance_pairs["score1"]
            self.score2 = variance_pairs["score2"]
        else:
            with open(file_path, "rb") as f:
                self.trajectory_pairs = pickle.load(f)

            for trajectory_pair in self.trajectory_pairs:
                trajectory1_flat = [
                    [item for sublist in trajectory_pair[0] for item in sublist]
                ]
                trajectory2_flat = [
                    [item for sublist in trajectory_pair[1] for item in sublist]
                ]
                self.first_trajectories.append(trajectory1_flat)
                self.second_trajectories.append(trajectory2_flat)
                self.labels.append(trajectory_pair[2])
                self.score1.append(trajectory_pair[3])
                self.score2.append(trajectory_pair[4])

            self.first_trajectories = torch.tensor(
                self.first_trajectories, dtype=torch.float32
            )
            self.second_trajectories = torch.tensor(
                self.second_trajectories, dtype=torch.float32
            )

            self.labels = torch.tensor(self.labels, dtype=torch.float32)
            self.score1 = torch.tensor(self.score1, dtype=torch.float32)
            self.score2 = torch.tensor(self.score2, dtype=torch.float32)

        if preload:
            self.first_trajectories, self.second_trajectories, self.labels, self.score1, self.score2 = load_tensors([self.first_trajectories, self.second_trajectories, self.labels, self.score1, self.score2])

    def __getitem__(self, idx):
        traj1 = self.first_trajectories[idx]
        traj2 = self.second_trajectories[idx]
        preference = self.labels[idx]
        score1 = self.score1[idx]
        score2 = self.score2[idx]
        return traj1, traj2, preference, score1, score2

    def __len__(self):
        return len(self.labels)


def bradley_terry_model(r1, r2):
    exp_r1 = torch.exp(r1)
    exp_r2 = torch.exp(r2)
    probability = exp_r1 / (exp_r1 + exp_r2)
    return probability.squeeze()


def preference_loss(predicted_probabilities, true_preferences):
    return F.binary_cross_entropy(predicted_probabilities, true_preferences)


def pick_highest_entropy_dataset(epoch, ensemble, train_dataset, subset_size, preload=True):
    def get_variance(traj1, traj2):
        if not preload:
            traj1 = traj1.to(device)
            traj2 = traj2.to(device)
        model_probabilities = []
        for model in ensemble.model_list:
            model_probabilities.append(bradley_terry_model(model(traj1), model(traj2)))
        variance = torch.var(torch.stack(model_probabilities)).item()
        return variance

    new_train_size = int(subset_size * train_ratio)

    if epoch == 0:
        train_sampler, _ = random_split(
            train_dataset, [new_train_size, len(train_dataset) - new_train_size]
        )
    else:
        pairs_w_variance = []
        for idx, (traj1, traj2, label, score1, score2) in enumerate(
            train_dataset.dataset
        ):
            pair_variance = get_variance(traj1, traj2)
            pairs_w_variance.append((idx, pair_variance))

        pairs_w_variance = sorted(pairs_w_variance, key=lambda x: x[1], reverse=True)

        top_indices = [item[0] for item in pairs_w_variance[:new_train_size]]

        variance_pairs = {
            "first_trajectories": [],
            "second_trajectories": [],
            "labels": [],
            "score1": [],
            "score2": [],
        }

        for index in top_indices:
            (traj1, traj2, label, score1, score2) = train_dataset.dataset[index]
            variance_pairs["first_trajectories"].append(traj1)
            variance_pairs["second_trajectories"].append(traj2)
            variance_pairs["labels"].append(label)
            variance_pairs["score1"].append(score1)
            variance_pairs["score2"].append(score2)

        train_sampler = TrajectoryDataset("", variance_pairs, preload)
    return train_sampler


def distribute_data(train_subset, batch_size, num_models, preload=False):
    pin_mem = not preload
    train_sizes = [len(train_subset) // num_models for _ in range(num_models)]
    print("TRAIN_SIZES:", train_sizes)
    first_train_size = len(train_subset) - num_models * (
        len(train_subset) // num_models
    )
    train_sizes[0] += first_train_size
    train_datasets = random_split(train_subset, train_sizes)
    train_dataloaders = [
        DataLoader(
            train_datasets[i],
            batch_size=train_sizes[i] if train_sizes[i] < batch_size else batch_size,
            shuffle=True,
            pin_memory=pin_mem,
        )
        for i in range(num_models)
    ]
    return train_dataloaders


def prepare_single_trajectory(trajectory, max_length=2):
    def truncate(trajectory, max_length):
        if len(trajectory) > max_length:
            return trajectory[-max_length:]
        return trajectory

    trajectory_flat = [
        item for sublist in truncate(trajectory, max_length) for item in sublist
    ]

    # Convert to tensor and add an extra dimension
    trajectory_tensor = torch.tensor(trajectory_flat, dtype=torch.float32).to(device)

    return trajectory_tensor


def calculate_accuracy(predicted_probabilities, true_preferences):
    predicted_preferences = (predicted_probabilities > 0.5).float()
    correct_predictions = (predicted_preferences == true_preferences).float().sum()
    accuracy = correct_predictions / true_preferences.size(0)
    return accuracy.item()


def calculate_adjusted_accuracy(
    predicted_probabilities, true_preferences, first_rewards, second_rewards
):
    differing_rewards_mask = first_rewards != second_rewards
    filtered_predicted_probabilities = predicted_probabilities[differing_rewards_mask]
    filtered_true_preferences = true_preferences[differing_rewards_mask]
    if filtered_true_preferences.size(0) == 0:
        return 0.0
    predicted_preferences = (filtered_predicted_probabilities > 0.5).float()
    correct_predictions = (
        (predicted_preferences == filtered_true_preferences).float().sum()
    )
    adjusted_accuracy = correct_predictions / filtered_true_preferences.size(0)
    return adjusted_accuracy.item()


def train_ensemble(
    ensemble, epochs, swaps, optimizer, batch_size, trajectory_path, return_stat=None, preload=False
):
    n_models = len(ensemble.model_list)
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs
    )

    pin_mem = not preload 

    full_dataset = TrajectoryDataset(trajectory_path, None, preload)
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=val_size if val_size < batch_size else batch_size,
        shuffle=False,
        pin_memory=pin_mem,
    )

    training_losses = []
    training_accuracies = []
    adjusted_training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    adjusted_validation_accuracies = []

    best_loss = float("inf")
    best_val_losses = [float("inf") for _ in range(n_models)]

    global ensemble_path
    ensemble_path = (
        models_path
        + f"ensemble_{epochs}_epochs_{dataset_size}_pairs_{rules.NUMBER_OF_RULES}_rules/"
    )
    os.makedirs(ensemble_path, exist_ok=True)

    epoch = 0
    while epoch < epochs:
        if epoch % (epochs // swaps) == 0:
            print("SWAP:", epoch)
            train_subset = pick_highest_entropy_dataset(
                epoch, ensemble, train_dataset, math.ceil(dataset_size / (swaps + 1)), preload
            )
            training_dataloaders = distribute_data(train_subset, batch_size, n_models, preload)

        loss_across_models = 0
        acc_across_models = 0
        adjusted_acc_across_models = 0
        val_loss_across_models = 0
        val_acc_across_models = 0
        adjusted_val_acc_across_models = 0

        for i in range(n_models):
            train_size = len(training_dataloaders[i].dataset)
            val_size = len(validation_dataloader.dataset)

            model = ensemble.model_list[i]
            model = model.to(device)
            model.train()

            model_loss = 0.0
            model_acc = 0.0
            model_adjusted_acc = 0.0

            for (
                batch_traj1,
                batch_traj2,
                batch_true_pref,
                batch_reward1,
                batch_reward2,
            ) in training_dataloaders[i]:
                if not preload:
                    batch_traj1, batch_traj2, batch_true_pref, batch_reward1, batch_reward2 = load_tensors([batch_traj1, batch_traj2, batch_true_pref, batch_reward1, batch_reward2])
                
                combined_batch = torch.cat([batch_traj1, batch_traj2], dim=0)
                combined_rewards = model(combined_batch)
                rewards1, rewards2 = torch.split(combined_rewards, [batch_traj1.shape[0], batch_traj2.shape[0]], dim=0)

                predicted_probabilities = bradley_terry_model(rewards1, rewards2)
                model_loss += preference_loss(predicted_probabilities, batch_true_pref) * batch_true_pref.size(0)
                model_acc += calculate_accuracy(predicted_probabilities, batch_true_pref) * batch_true_pref.size(0)
                model_adjusted_acc += calculate_adjusted_accuracy(
                    predicted_probabilities,
                    batch_true_pref,
                    batch_reward1,
                    batch_reward2,
                ) * batch_true_pref.size(0)

            model_loss /= train_size
            model_acc /= train_size
            model_adjusted_acc /= train_size
            loss_across_models += model_loss
            acc_across_models += model_acc
            adjusted_acc_across_models += model_adjusted_acc

            if epochs % 10 == 0:
                val_loss = 0.0
                val_acc = 0.0
                val_adjusted_acc = 0.0

                # Validation steps and logging
                model.eval()
                with torch.no_grad():
                    for (
                        validation_traj1,
                        validation_traj2,
                        validation_true_pref,
                        validation_reward1,
                        validation_reward2,
                    ) in validation_dataloader:
                        if not preload:
                            validation_traj1, validation_traj2, validation_true_pref, validation_reward1, validation_reward2 = load_tensors([validation_traj1, validation_traj2, validation_true_pref, validation_reward1, validation_reward2])
                        combined_val_batch = torch.cat([validation_traj1, validation_traj2], dim=0)
                        combined_val_rewards = model(combined_val_batch)
                        validation_rewards1, validation_rewards2 = torch.split(combined_val_rewards, [validation_traj1.shape[0], validation_traj2.shape[0]], dim=0)
                        
                        validation_predicted_probabilities = bradley_terry_model(
                            validation_rewards1, validation_rewards2
                        )
                        val_loss += preference_loss(
                            validation_predicted_probabilities, validation_true_pref
                        ) * validation_true_pref.size(0)
                        val_acc += calculate_accuracy(
                            validation_predicted_probabilities, validation_true_pref
                        ) * validation_true_pref.size(0)
                        val_adjusted_acc += calculate_adjusted_accuracy(
                            validation_predicted_probabilities,
                            validation_true_pref,
                            validation_reward1,
                            validation_reward2,
                        ) * validation_true_pref.size(0)

                val_loss /= val_size
                val_acc /= val_size
                val_adjusted_acc /= val_size
                val_loss_across_models += val_loss
                val_acc_across_models += val_acc
                adjusted_val_acc_across_models += val_adjusted_acc

                if val_loss < best_val_losses[i]:
                    best_val_losses[i] = val_loss
                    torch.save(
                        ensemble.model_list[i].state_dict(),
                        ensemble_path + f"model_{epochs}_epochs_{dataset_size}_pairs_{rules.NUMBER_OF_RULES}_rules_{i}.pth",
                    )
                    print(f"MODEL {i} SAVED AT EPOCH: {epoch}")

    
        avg_train_loss = loss_across_models / n_models
        avg_train_acc = acc_across_models / n_models
        avg_adjusted_train_acc = adjusted_acc_across_models / n_models
        best_loss = min(best_loss, avg_train_loss)

        training_losses.append(avg_train_loss.item())
        training_accuracies.append(avg_train_acc)
        adjusted_training_accuracies.append(avg_adjusted_train_acc)

        if epoch % 10 == 0:
            avg_val_loss = val_loss_across_models / n_models
            avg_val_acc = val_acc_across_models / n_models
            avg_adjusted_val_acc = adjusted_val_acc_across_models / n_models

            validation_losses.append(avg_val_loss.item())
            validation_accuracies.append(avg_val_acc)
            adjusted_validation_accuracies.append(avg_adjusted_val_acc)

            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}, Train Acc: {avg_train_acc} (adjusted: {avg_adjusted_train_acc}), Val Acc: {avg_val_acc} (adjusted: {avg_adjusted_val_acc})"
            )
        epoch += 1

        optimizer.zero_grad()
        avg_train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    try:
        global figure_path
        if isinstance(training_losses, torch.Tensor):
            print("training losses was a tensor")
            training_losses = training_losses.cpu().numpy()
        if isinstance(validation_losses, torch.Tensor):
            validation_losses = validation_losses.cpu().numpy()
        plt.figure()
        plt.plot(training_losses, label="Train Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{figure_path}loss.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(training_accuracies, label="Train Accuracy")
        plt.plot(validation_accuracies, label="Validation Accuracy")
        plt.plot(adjusted_training_accuracies, label="Adjusted Training Accuracy")
        plt.plot(adjusted_validation_accuracies, label="Adjusted Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{figure_path}accuracy.png", dpi=600)
        plt.close()
    except:
        print("Issue when plotting training and validation loss (they are tensors perhaps??)")

    training_output = best_loss
    if return_stat == "acc":
        training_output = {
            "final_training_acc": training_accuracies[-1],
            "final_validation_acc": validation_accuracies[-1],
            "final_adjusted_training_acc": adjusted_training_accuracies[-1],
            "final_adjusted_validation_acc": adjusted_validation_accuracies[-1],
        }
    print("Training over, returning:", training_output)
    return training_output


def train_model(
    file_path,
    net,
    epochs=1000,
    optimizer=None,
    batch_size=256,
    model_path="best.pth",
    return_stat=None,
    preload=True
):
    print("BATCH_SIZE:", batch_size, "| file path:", file_path, "| epochs:", epochs)
    wandb.init(project="Micro Preference")
    wandb.watch(net, log="all")

    # Create the dataset
    full_dataset = TrajectoryDataset(file_path, None, preload)

    global n_pairs
    n_pairs = len(full_dataset)

    # Define the split ratio
    train_ratio = 0.8
    # val_ratio = 0.2
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    pin_mem = not preload

    # Initialize Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_size if train_size < batch_size else batch_size,
        shuffle=True,
        pin_memory=pin_mem,
    )
    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=val_size if val_size < batch_size else batch_size,
        shuffle=False,
        pin_memory=pin_mem,
    )

    if batch_size > train_size:
        batch_size = train_size

    best_loss = np.inf
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    adjusted_training_accuracies = []
    adjusted_validation_accuracies = []

    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs
    )

    epoch = 0
    try:
        while epoch < epochs:
            # Training
            net.train()
            total_loss = 0.0
            total_accuracy = 0.0
            total_adjusted_accuracy = 0.0
            total_probability = 0.0

            for (
                batch_traj1,
                batch_traj2,
                batch_true_pref,
                batch_score1,
                batch_score2,
            ) in train_dataloader:
                if not preload:
                    batch_traj1, batch_traj2, batch_true_pref, batch_score1, batch_score2 = load_tensors([batch_traj1, batch_traj2, batch_true_pref, batch_score1, batch_score2])

                combined_batch = torch.cat([batch_traj1, batch_traj2], dim=0)
                combined_rewards = net(combined_batch)
                rewards1, rewards2 = torch.split(combined_rewards, [batch_traj1.shape[0], batch_traj2.shape[0]], dim=0)

                predicted_probabilities = bradley_terry_model(rewards1, rewards2)
                total_probability += predicted_probabilities.sum().item()

                loss = preference_loss(predicted_probabilities, batch_true_pref)
                accuracy = calculate_accuracy(predicted_probabilities, batch_true_pref)
                adjusted_accuracy = calculate_adjusted_accuracy(
                    predicted_probabilities, batch_true_pref, batch_score1, batch_score2
                )

                total_loss += loss.item() * batch_true_pref.size(0)
                total_accuracy += accuracy * batch_true_pref.size(0)
                total_adjusted_accuracy += adjusted_accuracy * batch_true_pref.size(0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            average_training_loss = total_loss / train_size
            training_losses.append(average_training_loss)

            average_training_accuracy = total_accuracy / train_size
            average_adjusted_training_accuracy = total_adjusted_accuracy / train_size
            training_accuracies.append(average_training_accuracy)
            adjusted_training_accuracies.append(average_adjusted_training_accuracy)

            # Validation
            if epoch % 10 == 0:
                net.eval()
                total_validation_loss = 0.0
                total_validation_accuracy = 0.0
                total_adjusted_validation_accuracy = 0.0
                with torch.no_grad():
                    for (
                        validation_traj1,
                        validation_traj2,
                        validation_true_pref,
                        validation_score1,
                        validation_score2,
                    ) in validation_dataloader:
                        if not preload:
                            validation_traj1, validation_traj2, validation_true_pref, validation_score1, validation_score2 = load_tensors([validation_traj1, validation_traj2, validation_true_pref, validation_score1, validation_score2])

                        combined_val_batch = torch.cat([validation_traj1, validation_traj2], dim=0)
                        combined_val_rewards = net(combined_val_batch)
                        validation_rewards1, validation_rewards2 = torch.split(combined_val_rewards, [validation_traj1.shape[0], validation_traj2.shape[0]], dim=0)
                        
                        validation_predicted_probabilities = bradley_terry_model(
                            validation_rewards1, validation_rewards2
                        )                    

                        validation_loss = preference_loss(
                            validation_predicted_probabilities, validation_true_pref
                        )
                        validation_accuracy = calculate_accuracy(
                            validation_predicted_probabilities, validation_true_pref
                        )
                        adjusted_validation_accuracy = calculate_adjusted_accuracy(
                            validation_predicted_probabilities,
                            validation_true_pref,
                            validation_score1,
                            validation_score2,
                        )

                        total_validation_loss += validation_loss * validation_true_pref.size(0)
                        total_validation_accuracy += validation_accuracy * validation_true_pref.size(0)
                        total_adjusted_validation_accuracy += adjusted_validation_accuracy * validation_true_pref.size(0)

                average_validation_loss = total_validation_loss / val_size 
                if average_validation_loss < best_loss:
                    best_loss = average_validation_loss
                    torch.save(
                        net.state_dict(),
                        model_path
                        + f"_{10 ** round(math.log10(n_pairs))}_pairs_{rules.NUMBER_OF_RULES}_rules.pth",
                    )
                    # print("MODEL SAVED AT EPOCH:", epoch)
                average_validation_accuracy = total_validation_accuracy / val_size
                average_adjusted_validation_accuracy = (
                    total_adjusted_validation_accuracy / val_size
                )
                validation_losses.append(average_validation_loss.item())
                validation_accuracies.append(average_validation_accuracy)
                adjusted_validation_accuracies.append(average_adjusted_validation_accuracy)

                wandb.log(
                    {
                        "Train Loss": average_training_loss,
                        "Validation Loss": average_validation_loss.item(),
                        "Train Accuracy": average_training_accuracy,
                        "Validation Accuracy": average_validation_accuracy,
                        "Adjusted Train Accuracy": average_adjusted_training_accuracy,
                        "Adjusted Validation Accuracy": average_adjusted_validation_accuracy,
                    },
                    step=epoch,
                )

                print(
                    f"Epoch {epoch}/{epochs}, Train Loss: {average_training_loss}, Val Loss: {average_validation_loss.item()}, Train Acc: {average_training_accuracy} (adjusted: {average_adjusted_training_accuracy}), Val Acc: {average_validation_accuracy} (adjusted: {average_adjusted_validation_accuracy})"
                )
            epoch += 1
    except Exception as e:
        torch.save(
            net.state_dict(),
            model_path + f"_{n_pairs}_pairs_{rules.NUMBER_OF_RULES}_rules.pth",
        )
        print("EXCEPTION CAUGHT AND MODEL SAVED AT EPOCH:", epoch)
        print(e)

    global figure_path
    plt.figure()
    plt.plot(training_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{figure_path}loss.png", dpi=600)
    plt.close()

    plt.figure()
    plt.plot(training_accuracies, label="Train Accuracy")
    plt.plot(validation_accuracies, label="Validation Accuracy")
    plt.plot(adjusted_training_accuracies, label="Adjusted Train Accuracy")
    plt.plot(adjusted_validation_accuracies, label="Adjusted Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{figure_path}accuracy.png", dpi=600)
    plt.close()

    training_output = best_loss
    if return_stat == "acc":
        training_output = {
            "final_training_acc": training_accuracies[-1],
            "final_validation_acc": validation_accuracies[-1],
            "final_adjusted_training_acc": adjusted_training_accuracies[-1],
            "final_adjusted_validation_acc": adjusted_validation_accuracies[-1],
        }
    print("Training over, returning:", training_output)
    return training_output


def train_reward_function(
    trajectories_file_path,
    epochs,
    parameters_path=None,
    use_ensemble=False,
    figure_folder_name=None,
    return_stat=None,
    save_at_end=True,
):
    torch.cuda.empty_cache()
    training_output_stat = None
    input_size = INPUT_SIZE
    if use_ensemble:
        ensemble_path = models_path + f"ensemble_{epochs}/"
        os.makedirs(ensemble_path, exist_ok=True)

    if figure_folder_name:
        global figure_path
        figure_path = figure_folder_name + "/"
        os.makedirs(figure_path, exist_ok=True)

    # OPTUNA
    if not parameters_path:
        print("RUNNING WITH OPTUNA:")
        global study
        study = optuna.create_study(direction="minimize")
        study.set_user_attr("file_path", trajectories_file_path)
        study.set_user_attr("epochs", epochs)
        if use_ensemble:
            study.optimize(ensemble_objective, n_trials=OPTUNA_N_TRIALS)
        else:
            study.optimize(objective, n_trials=OPTUNA_N_TRIALS)

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
            models_path + torch.load(f"best_model_trial_{best_trial.number}.pth")
        )
        torch.save(
            best_model.state_dict(),
            models_path + f"best_model_{best_trial.params['hidden_size']}.pth",
        )

        # Load the best models
        best_models = [
            TrajectoryRewardNet(input_size, best_trial.params["hidden_size"]).to(device)
            for _ in range(ENSEMBLE_SIZE)
        ]
        for i in range(ENSEMBLE_SIZE):
            best_models[i].load_state_dict(
                torch.load(
                    models_path
                    + f"best_ensemble_{best_trial.number}/"
                    + f"model_{epochs}_{i}.pth",
                )
            )
        for i in range(ENSEMBLE_SIZE):
            torch.save(
                best_model.state_dict(),
                models_path
                + "best_ensemble/"
                + f"best_model_{best_trial.params['hidden_size']}_{i}.pth",
            )

        # Delete saved hyperparameter trials
        for file in glob.glob("best_model_trial_*.pth"):
            os.remove(file)

        for file in glob.glob("best_ensemble/*"):
            os.remove(file)

    # NO OPTUNA
    else:
        print("RUNNING WITHOUT OPTUNA:")
        with open(parameters_path, "r") as file:
            data = yaml.safe_load(file)
            hidden_size = data["hidden_size"]
            learning_rate = data["learning_rate"]
            weight_decay = data["weight_decay"]
            dropout_prob = data["dropout_prob"]
            swaps = data["swaps"]
            batch_size = data["batch_size"]

            if use_ensemble:
                ensemble = Ensemble(
                    input_size, ENSEMBLE_SIZE, None, hidden_size, dropout_prob
                ).to(device)
                # net = TrajectoryRewardNet(input_size, hidden_size, dropout_prob).to(device)
                for param in ensemble.parameters():
                    if len(param.shape) > 1:
                        nn.init.xavier_uniform_(
                            param, gain=nn.init.calculate_gain("relu")
                        )
                optimizer = torch.optim.Adam(
                    ensemble.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                training_output_stat = train_ensemble(
                    ensemble,
                    epochs,
                    swaps,
                    optimizer,
                    batch_size,
                    trajectories_file_path,
                    return_stat,
                    True,
                )
                if save_at_end:
                    for i in range(len(ensemble.model_list)):
                        torch.save(
                            ensemble.model_list[i].state_dict(),
                            ensemble_path + f"model_{epochs}_{i}.pth",
                        )
            else:
                net = TrajectoryRewardNet(input_size, hidden_size, dropout_prob).to(
                    device
                )
                for param in net.parameters():
                    if len(param.shape) > 1:
                        nn.init.xavier_uniform_(
                            param, gain=nn.init.calculate_gain("relu")
                        )
                optimizer = torch.optim.Adam(
                    net.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                training_output_stat = train_model(
                    file_path=trajectories_file_path,
                    net=net,
                    epochs=epochs,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    model_path=models_path + f"model_{epochs}_epochs",
                    return_stat=return_stat,
                    preload=True,
                )
                if save_at_end:
                    global n_pairs
                    torch.save(
                        net.state_dict(),
                        models_path
                        + f"model_{epochs}_epochs_{n_pairs}_pairs_{rules.NUMBER_OF_RULES}_rules.pth",
                    )
        return training_output_stat


def objective(trial):
    input_size = INPUT_SIZE
    hidden_size = trial.suggest_int("hidden_size", 64, 1024)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3)
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.8)
    batch_size = trial.suggest_int("batch_size", 3333, 20000)
    swaps=trial.suggest_int("swaps", 3, 3)

    net = TrajectoryRewardNet(input_size, hidden_size, dropout_prob).to(device)
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
        torch.save(
            net.state_dict(), models_path + f"best_model_trial_{trial.number}.pth"
        )

    return best_loss


def ensemble_objective(trial):
    input_size = INPUT_SIZE
    hidden_size = trial.suggest_int("hidden_size", 64, 1024)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3)
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.8)
    batch_size = trial.suggest_int("batch_size", 5000, 20000)
    swaps = trial.suggest_int("swaps", 0, 10)

    ensemble = Ensemble(input_size, 3, hidden_size, dropout_prob).to(device)
    # net = TrajectoryRewardNet(input_size, hidden_size, dropout_prob).to(device)
    for param in ensemble.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))
    optimizer = torch.optim.Adam(
        ensemble.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_loss = train_ensemble(
        ensemble, epochs, swaps, optimizer, study.user_attrs["file_path"], batch_size=batch_size
    )

    # Save the best model parameters
    if trial.should_prune():
        raise optuna.TrialPruned()
    else:
        # Save model state with trial number to avoid overwrite
        for i, model in enumerate(ensemble.model_list):
            torch.save(
                model.state_dict(),
                models_path
                + f"best_ensemble_{trial.number}/"
                + f"model_{epochs}_{i}.pth",
            )

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
    parse.add_argument("-l", "--labels", type=int, help="Number of labels created")
    parse.add_argument(
        "--ensemble", action="store_true", help="Train an ensemble of 3 predictors"
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
        epochs = 50

    train_reward_function(data_path, epochs, args.parameters, args.ensemble)

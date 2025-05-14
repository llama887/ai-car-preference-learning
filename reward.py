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
import random

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
            num_workers=0,
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
        num_workers=0,
    )

    # --- Early Stopping Initialization ---
    patience = 10 # Hardcoded patience
    validation_frequency = 50 # Check validation every 50 epochs
    best_avg_val_loss = float('inf')
    epochs_no_improve = 0 # Counts validation checks without improvement
    best_ensemble_states = None # Store state dicts for all models in the best ensemble
    # --- End Early Stopping Initialization ---

    training_losses = []
    training_accuracies = []
    adjusted_training_accuracies = []
    validation_losses = [] # Will store losses only from validation epochs
    validation_accuracies = [] # Will store accuracies only from validation epochs
    adjusted_validation_accuracies = [] # Will store adj accuracies only from validation epochs

    global ensemble_path
    ensemble_path = (
        models_path
        + f"ensemble_{epochs}_epochs_{dataset_size}_pairs_{rules.NUMBER_OF_RULES}_rules/"
    )
    os.makedirs(ensemble_path, exist_ok=True)

    epoch = 0
    try:
        while epoch < epochs:
            if epoch % (epochs // swaps) == 0 and swaps > 0: # Avoid division by zero if swaps is 0
                print("SWAP:", epoch)
                train_subset = pick_highest_entropy_dataset(
                    epoch, ensemble, train_dataset, math.ceil(dataset_size / (swaps + 1)), preload
                )
                training_dataloaders = distribute_data(train_subset, batch_size, n_models, preload)
            elif epoch == 0: # Initial data distribution if no swaps or first epoch
                 train_subset = pick_highest_entropy_dataset(
                    epoch, ensemble, train_dataset, math.ceil(dataset_size / (swaps + 1)), preload
                )
                 training_dataloaders = distribute_data(train_subset, batch_size, n_models, preload)


            loss_across_models = 0
            acc_across_models = 0
            adjusted_acc_across_models = 0

            # --- Training Phase ---
            for i in range(n_models):
                train_size_model = len(training_dataloaders[i].dataset) # Use actual size for this model's subset
                val_size_model = len(validation_dataloader.dataset) # Total validation size

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
                    loss = preference_loss(predicted_probabilities, batch_true_pref) # Calculate loss for backprop

                    model_loss += loss.item() * batch_true_pref.size(0) # Accumulate loss value
                    model_acc += calculate_accuracy(predicted_probabilities, batch_true_pref) * batch_true_pref.size(0)
                    model_adjusted_acc += calculate_adjusted_accuracy(
                        predicted_probabilities,
                        batch_true_pref,
                        batch_reward1,
                        batch_reward2,
                    ) * batch_true_pref.size(0)

                    # Accumulate gradients for the average loss later
                    (loss / n_models).backward() # Scale loss for averaging before backward pass

                model_loss /= train_size_model
                model_acc /= train_size_model
                model_adjusted_acc /= train_size_model
                loss_across_models += model_loss # Accumulate average loss for this model
                acc_across_models += model_acc
                adjusted_acc_across_models += model_adjusted_acc

            # Calculate and store average training metrics for the epoch
            avg_train_loss = loss_across_models / n_models # Average of averages
            avg_train_acc = acc_across_models / n_models
            avg_adjusted_train_acc = adjusted_acc_across_models / n_models
            training_losses.append(avg_train_loss) # Store scalar value
            training_accuracies.append(avg_train_acc)
            adjusted_training_accuracies.append(avg_adjusted_train_acc)

            # --- Validation Phase (every validation_frequency epochs) ---
            if epoch % validation_frequency == 0:
                val_loss_across_models = 0.0 # Reset for the current validation epoch
                val_acc_across_models = 0.0
                adjusted_val_acc_across_models = 0.0
                current_ensemble_val_loss = 0.0 # Accumulate total loss for averaging

                for i in range(n_models): # Validate each model
                    model = ensemble.model_list[i]
                    model.eval() # Set model to evaluation mode
                    model_val_loss = 0.0
                    model_val_acc = 0.0
                    model_val_adjusted_acc = 0.0

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
                            val_loss = preference_loss(
                                validation_predicted_probabilities, validation_true_pref
                            )
                            model_val_loss += val_loss.item() * validation_true_pref.size(0)
                            model_val_acc += calculate_accuracy(
                                validation_predicted_probabilities, validation_true_pref
                            ) * validation_true_pref.size(0)
                            model_val_adjusted_acc += calculate_adjusted_accuracy(
                                validation_predicted_probabilities,
                                validation_true_pref,
                                validation_reward1,
                                validation_reward2,
                            ) * validation_true_pref.size(0)

                    model_val_loss /= val_size # Average loss for this model on validation set
                    model_val_acc /= val_size
                    model_val_adjusted_acc /= val_size
                    val_loss_across_models += model_val_loss # Sum of average losses
                    val_acc_across_models += model_val_acc
                    adjusted_val_acc_across_models += model_val_adjusted_acc

                # --- Calculate Averages and Check Early Stopping ---
                avg_val_loss = val_loss_across_models / n_models # Average validation loss for the ensemble
                avg_val_acc = val_acc_across_models / n_models
                avg_adjusted_val_acc = adjusted_val_acc_across_models / n_models

                # Append validation metrics only when calculated
                validation_losses.append(avg_val_loss)
                validation_accuracies.append(avg_val_acc)
                adjusted_validation_accuracies.append(avg_adjusted_val_acc)

                print(
                     f"Epoch {epoch}/{epochs}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, Avg Train Acc: {avg_train_acc:.4f} (adjusted: {avg_adjusted_train_acc:.4f}), Avg Val Acc: {avg_val_acc:.4f} (adjusted: {avg_adjusted_val_acc:.4f})"
                )

                if avg_val_loss < best_avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    # Store the state dicts of all models in the current best ensemble
                    best_ensemble_states = [m.state_dict() for m in ensemble.model_list]
                    print(f"Epoch {epoch}: Average validation loss improved to {best_avg_val_loss:.4f}. Best ensemble state updated.")
                else:
                    epochs_no_improve += 1
                    print(f"Epoch {epoch}: Average validation loss ({avg_val_loss:.4f}) did not improve from best ({best_avg_val_loss:.4f}). Checks without improvement: {epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience: # Check against hardcoded patience
                    print(f"Early stopping triggered after {epoch} epochs ({epochs_no_improve} checks without improvement).")
                    break # Exit the training loop
                # --- End Early Stopping Check ---

            # --- Training Step ---
            optimizer.zero_grad() # Zero gradients before stepping
            # Gradients were already calculated and accumulated during the training phase loop
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # --- End Training Step ---

            epoch += 1 # Increment epoch
    except Exception as e:
        print(f"EXCEPTION CAUGHT during ensemble training at epoch {epoch}: {e}")
        # Save the *last* state before exception
        print("Saving last ensemble state before exception.")
        for i, model in enumerate(ensemble.model_list):
             save_path = ensemble_path + f"model_{epochs}_epochs_{dataset_size}_pairs_{rules.NUMBER_OF_RULES}_rules_{i}_EXCEPTION.pth"
             torch.save(model.state_dict(), save_path)
             print(f"Saved last state for model {i} to {save_path}")
        raise e # Re-raise the exception


    # --- Save the best ensemble after training ---
    if best_ensemble_states:
        print(f"Saving best ensemble found with avg validation loss: {best_avg_val_loss:.4f}")
        for i, state_dict in enumerate(best_ensemble_states):
            save_path = ensemble_path + f"model_{epochs}_epochs_{dataset_size}_pairs_{rules.NUMBER_OF_RULES}_rules_{i}.pth"
            torch.save(state_dict, save_path)
            print(f"Saved best state for model {i} to {save_path}")
    else:
        # If no validation improvement happened, save the last state of the ensemble
        print("No validation improvement recorded for ensemble average. Saving last ensemble state.")
        for i, model in enumerate(ensemble.model_list):
             save_path = ensemble_path + f"model_{epochs}_epochs_{dataset_size}_pairs_{rules.NUMBER_OF_RULES}_rules_{i}_LAST.pth"
             torch.save(model.state_dict(), save_path)
             print(f"Saved last state for model {i} to {save_path}")
    # --- End Save Best Ensemble ---


    try:
        global figure_path
        # Ensure lists are suitable for plotting (e.g., convert tensors if necessary)
        plot_train_losses = [l.item() if isinstance(l, torch.Tensor) else l for l in training_losses]
        plot_val_losses = [l.item() if isinstance(l, torch.Tensor) else l for l in validation_losses]
        plot_train_acc = [l.item() if isinstance(l, torch.Tensor) else l for l in training_accuracies]
        plot_val_acc = [l.item() if isinstance(l, torch.Tensor) else l for l in validation_accuracies]
        plot_adj_train_acc = [l.item() if isinstance(l, torch.Tensor) else l for l in adjusted_training_accuracies]
        plot_adj_val_acc = [l.item() if isinstance(l, torch.Tensor) else l for l in adjusted_validation_accuracies]

        # Create x-axis values for validation plots (sparse)
        validation_epochs = list(range(0, epoch, validation_frequency)) # Epochs where validation was run
        if epoch % validation_frequency == 0 and epoch > 0: # Include last epoch if validation ran
             pass # Already included by range
        elif epoch > 0 and len(validation_losses) > len(validation_epochs): # Handle early stopping case
             validation_epochs.append(epoch-1) # Add the epoch where it stopped


        plt.figure()
        plt.plot(plot_train_losses, label="Train Loss (per epoch)")
        # Plot validation loss if data exists
        if validation_epochs and plot_val_losses:
             # Ensure lengths match for plotting, potentially adjusting validation_epochs if needed due to early stop timing
             plot_epoch_count = min(len(validation_epochs), len(plot_val_losses))
             plt.plot(validation_epochs[:plot_epoch_count], plot_val_losses[:plot_epoch_count], label="Validation Loss (per check)", marker='o', linestyle='--')
        elif plot_val_losses:
             print(f"Could not plot ensemble validation loss: Mismatch between validation epochs ({len(validation_epochs)}) and recorded losses ({len(plot_val_losses)}).")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{figure_path}ensemble_loss.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(plot_train_acc, label="Train Accuracy (per epoch)")
        plt.plot(plot_adj_train_acc, label="Adjusted Training Accuracy (per epoch)")
        # Plot validation accuracy if data exists
        if validation_epochs and plot_val_acc:
             plot_epoch_count_acc = min(len(validation_epochs), len(plot_val_acc))
             plt.plot(validation_epochs[:plot_epoch_count_acc], plot_val_acc[:plot_epoch_count_acc], label="Validation Accuracy (per check)", marker='o', linestyle='--')
             # Plot adjusted validation accuracy if data exists and lengths match
             plot_epoch_count_adj_acc = min(len(validation_epochs), len(plot_adj_val_acc))
             if plot_adj_val_acc and plot_epoch_count_adj_acc > 0:
                 plt.plot(validation_epochs[:plot_epoch_count_adj_acc], plot_adj_val_acc[:plot_epoch_count_adj_acc], label="Adjusted Validation Accuracy (per check)", marker='x', linestyle=':')
             elif plot_adj_val_acc:
                 print(f"Could not plot ensemble adjusted validation accuracy: Mismatch between validation epochs ({len(validation_epochs)}) and recorded adjusted accuracies ({len(plot_adj_val_acc)}).")
        elif plot_val_acc:
             print(f"Could not plot ensemble validation accuracy: Mismatch between validation epochs ({len(validation_epochs)}) and recorded accuracies ({len(plot_val_acc)}).")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{figure_path}ensemble_accuracy.png", dpi=600)
        plt.close()
    except Exception as plot_err:
        print(f"Issue when plotting ensemble training/validation metrics: {plot_err}")

    # Return the best average validation loss achieved
    training_output = best_avg_val_loss
    if return_stat == "acc":
        # Accuracies are from the *last calculated validation epoch*
         training_output = {
            "final_training_acc": training_accuracies[-1] if training_accuracies else 0,
            "final_validation_acc": validation_accuracies[-1] if validation_accuracies else 0,
            "final_adjusted_training_acc": adjusted_training_accuracies[-1] if adjusted_training_accuracies else 0,
            "final_adjusted_validation_acc": adjusted_validation_accuracies[-1] if adjusted_validation_accuracies else 0,
            "best_avg_validation_loss": best_avg_val_loss
        }
    print("Ensemble training over, returning:", training_output)
    return training_output


def train_model_without_dataloader(
    file_path,
    net,
    epochs=1000,
    optimizer=None,
    batch_size=256,
    base_model_path="best_no_loader.pth",
    return_stat=None,
    preload=True
):
    print(f"Training without DataLoaders | file path: {file_path} | epochs: {epochs}")
    wandb.init(project="Micro Preference No Dataloader")
    wandb.watch(net, log="all")

    # Create the dataset
    full_dataset = TrajectoryDataset(file_path, None, preload)

    global n_pairs
    n_pairs = len(full_dataset)

    # Define the split ratio
    train_ratio = 0.8
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # --- Early Stopping Initialization ---
    patience = 10 
    validation_frequency = 50
    best_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None
    # --- End Early Stopping Initialization ---

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
            # --- Training Phase ---
            net.train()
            total_loss = 0.0
            total_accuracy = 0.0
            total_adjusted_accuracy = 0.0
            # total_probability = 0.0 # This wasn't used, removed

            optimizer.zero_grad() # Zero gradients at the start of the epoch

            for idx in range(math.ceil(train_size / batch_size)):
                start_idx = idx * batch_size
                end_idx = min(start_idx + batch_size, train_size)
                
                batch_traj1 = train_dataset.dataset.first_trajectories[start_idx:end_idx]
                batch_traj2 = train_dataset.dataset.second_trajectories[start_idx:end_idx]
                batch_true_pref = train_dataset.dataset.labels[start_idx:end_idx]
                batch_score1 = train_dataset.dataset.score1[start_idx:end_idx]
                batch_score2 = train_dataset.dataset.score2[start_idx:end_idx]

                if not preload:
                    batch_traj1, batch_traj2, batch_true_pref, batch_score1, batch_score2 = load_tensors([batch_traj1, batch_traj2, batch_true_pref, batch_score1, batch_score2])

                combined_batch = torch.cat([batch_traj1, batch_traj2], dim=0)
                combined_rewards = net(combined_batch)
                rewards1, rewards2 = torch.split(combined_rewards, [batch_traj1.shape[0], batch_traj2.shape[0]], dim=0)

                predicted_probabilities = bradley_terry_model(rewards1, rewards2)
                # total_probability += predicted_probabilities.sum().item() # This wasn't used

                loss = preference_loss(predicted_probabilities, batch_true_pref)
                accuracy = calculate_accuracy(predicted_probabilities, batch_true_pref)
                adjusted_accuracy = calculate_adjusted_accuracy(
                    predicted_probabilities, batch_true_pref, batch_score1, batch_score2
                )

                total_loss += loss.item() * batch_true_pref.size(0)
                total_accuracy += accuracy * batch_true_pref.size(0)
                total_adjusted_accuracy += adjusted_accuracy * batch_true_pref.size(0)

                loss.backward() # Calculate gradients

            # --- Training Step ---
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) # Clip gradients after accumulating for the epoch
            optimizer.step() # Update weights
            scheduler.step() # Update learning rate
            # --- End Training Step ---


            # Calculate and store average training metrics for the epoch
            average_training_loss = total_loss / train_size
            average_training_accuracy = total_accuracy / train_size
            average_adjusted_training_accuracy = total_adjusted_accuracy / train_size
            training_losses.append(average_training_loss)
            training_accuracies.append(average_training_accuracy)
            adjusted_training_accuracies.append(average_adjusted_training_accuracy)


            # --- Validation Phase (every validation_frequency epochs) ---
            if epoch % validation_frequency == 0:
                net.eval()
                total_validation_loss = 0.0
                total_validation_accuracy = 0.0
                total_adjusted_validation_accuracy = 0.0
                with torch.no_grad():
                    for idx in range(math.ceil(val_size / batch_size)):
                        start_idx = idx * batch_size
                        end_idx = min(start_idx + batch_size, val_size)

                        validation_traj1 = val_dataset.dataset.first_trajectories[start_idx:end_idx]
                        validation_traj2 = val_dataset.dataset.second_trajectories[start_idx:end_idx]
                        validation_true_pref = val_dataset.dataset.labels[start_idx:end_idx]
                        validation_score1 = val_dataset.dataset.score1[start_idx:end_idx]
                        validation_score2 = val_dataset.dataset.score2[start_idx:end_idx]
    
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

                        total_validation_loss += validation_loss.item() * validation_true_pref.size(0) # Use .item()
                        total_validation_accuracy += validation_accuracy * validation_true_pref.size(0)
                        total_adjusted_validation_accuracy += adjusted_validation_accuracy * validation_true_pref.size(0)

                average_validation_loss = total_validation_loss / val_size
                average_validation_accuracy = total_validation_accuracy / val_size
                average_adjusted_validation_accuracy = (
                    total_adjusted_validation_accuracy / val_size
                )

                # Append validation metrics only when calculated
                validation_losses.append(average_validation_loss)
                validation_accuracies.append(average_validation_accuracy)
                adjusted_validation_accuracies.append(average_adjusted_validation_accuracy)

                wandb.log(
                    {
                        # Log training loss from the *current* epoch
                        "Train Loss": average_training_loss,
                        "Validation Loss": average_validation_loss,
                        "Train Accuracy": average_training_accuracy,
                        "Validation Accuracy": average_validation_accuracy,
                        "Adjusted Train Accuracy": average_adjusted_training_accuracy,
                        "Adjusted Validation Accuracy": average_adjusted_validation_accuracy,
                    },
                    step=epoch,
                )

                print(
                    f"Epoch {epoch}/{epochs}, Train Loss: {average_training_loss:.4f}, Val Loss: {average_validation_loss:.4f}, Train Acc: {average_training_accuracy:.4f} (adjusted: {average_adjusted_training_accuracy:.4f}), Val Acc: {average_validation_accuracy:.4f} (adjusted: {average_adjusted_validation_accuracy:.4f})"
                )

                # --- Early Stopping Check ---
                if average_validation_loss < best_loss:
                    best_loss = average_validation_loss
                    epochs_no_improve = 0 # Reset counter
                    # Store the state dict of the best model found so far
                    best_model_state = net.state_dict()
                    print(f"Epoch {epoch}: Validation loss improved to {best_loss:.4f}. Best model state updated.")
                else:
                    # Increment counter only when validation is checked and no improvement is found
                    epochs_no_improve += 1
                    print(f"Epoch {epoch}: Validation loss ({average_validation_loss:.4f}) did not improve from best ({best_loss:.4f}). Checks without improvement: {epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience: # Check against hardcoded patience
                    print(f"Early stopping triggered after {epoch} epochs ({epochs_no_improve} checks without improvement).")
                    break # Exit the training loop
                # --- End Early Stopping Check ---

            # --- End Validation Block ---

            epoch += 1
    except Exception as e:
        # Save the best state found before exception, if available
        save_path = base_model_path + f"_EXCEPTION_{n_pairs}_pairs_{rules.NUMBER_OF_RULES}_rules.pth"
        if best_model_state:
            print("Saving best model state found before exception.")
            torch.save(best_model_state, save_path)
        else:
            print("Saving current model state before exception (no best state recorded).")
            torch.save(net.state_dict(), save_path)
        print(f"EXCEPTION CAUGHT during model training at epoch {epoch}: {e}")
        raise e

    # --- Save the best model after training completes (or early stopping) ---
    try:
        n_pairs_magnitude = 10 ** round(math.log10(n_pairs)) if n_pairs > 0 else 0
    except ValueError:
        n_pairs_magnitude = 0

    if best_model_state:
        final_save_path = base_model_path + f"_{n_pairs_magnitude}_pairs_{rules.NUMBER_OF_RULES}_rules.pth"
        torch.save(best_model_state, final_save_path)
        print(f"Best model state saved to {final_save_path} with validation loss: {best_loss:.4f}")
    else:
        final_save_path = base_model_path + f"_{n_pairs_magnitude}_pairs_{rules.NUMBER_OF_RULES}_rules_LAST.pth"
        torch.save(net.state_dict(), final_save_path)
        print(f"No validation improvement recorded. Saving last model state to {final_save_path}")

    # Plot training metrics
    try:
        global figure_path
        # Create x-axis values for validation plots (sparse)
        validation_epochs = list(range(0, epoch, validation_frequency))
        if epoch % validation_frequency == 0 and epoch > 0:
            pass
        elif epoch > 0 and len(validation_losses) > len(validation_epochs):
            validation_epochs.append(epoch-1)

        plt.figure()
        plt.plot(training_losses, label="Train Loss (per epoch)")
        if validation_epochs and validation_losses:
            plot_epoch_count = min(len(validation_epochs), len(validation_losses))
            plt.plot(validation_epochs[:plot_epoch_count], validation_losses[:plot_epoch_count], 
                    label="Validation Loss (per check)", marker='o', linestyle='--')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{figure_path}loss_no_dataloader.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(training_accuracies, label="Train Accuracy (per epoch)")
        plt.plot(adjusted_training_accuracies, label="Adjusted Training Accuracy (per epoch)")
        if validation_epochs and validation_accuracies:
            plot_epoch_count_acc = min(len(validation_epochs), len(validation_accuracies))
            plt.plot(validation_epochs[:plot_epoch_count_acc], validation_accuracies[:plot_epoch_count_acc], 
                    label="Validation Accuracy (per check)", marker='o', linestyle='--')
            plot_epoch_count_adj_acc = min(len(validation_epochs), len(adjusted_validation_accuracies))
            if adjusted_validation_accuracies and plot_epoch_count_adj_acc > 0:
                plt.plot(validation_epochs[:plot_epoch_count_adj_acc], adjusted_validation_accuracies[:plot_epoch_count_adj_acc], 
                        label="Adjusted Validation Accuracy (per check)", marker='x', linestyle=':')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{figure_path}accuracy_no_dataloader.png", dpi=600)
        plt.close()
    except Exception as plot_err:
        print(f"Issue when plotting training/validation metrics: {plot_err}")

    # Return the best validation loss achieved
    training_output = best_loss
    if return_stat == "acc":
        training_output = {
            "final_training_acc": training_accuracies[-1] if training_accuracies else 0,
            "final_validation_acc": validation_accuracies[-1] if validation_accuracies else 0,
            "final_adjusted_training_acc": adjusted_training_accuracies[-1] if adjusted_training_accuracies else 0,
            "final_adjusted_validation_acc": adjusted_validation_accuracies[-1] if adjusted_validation_accuracies else 0,
            "best_validation_loss": best_loss
        }
    print("Training without DataLoaders over, returning:", training_output)
    return training_output


def train_model(
    file_path,
    net,
    epochs=1000,
    optimizer=None,
    batch_size=256,
    base_model_path="best.pth", # Base path for saving
    return_stat=None,
    preload=True
    # patience parameter removed
):
    print("BATCH_SIZE:", batch_size, "| file path:", file_path, "| epochs:", epochs) # Log parameters
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
        num_workers=0,
    )
    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=val_size if val_size < batch_size else batch_size,
        shuffle=False,
        pin_memory=pin_mem,
        num_workers=0,
    )

    if batch_size > train_size:
        print(f"Warning: batch_size ({batch_size}) > train_size ({train_size}). Setting batch_size to {train_size}")
        batch_size = train_size

    # --- Early Stopping Initialization ---
    patience = 10 # Hardcoded patience: stop after 10 validation checks without improvement
    validation_frequency = 50 # Perform validation check every 50 epochs
    best_loss = np.inf # Tracks the best validation loss
    epochs_no_improve = 0 # Counts consecutive validation checks without improvement
    best_model_state = None # Store the state dict of the best model found so far
    # --- End Early Stopping Initialization ---

    training_losses = []
    validation_losses = [] # Will store losses only from validation epochs
    training_accuracies = []
    validation_accuracies = [] # Will store accuracies only from validation epochs
    adjusted_training_accuracies = []
    adjusted_validation_accuracies = [] # Will store adj accuracies only from validation epochs

    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs
    )

    epoch = 0
    try:
        while epoch < epochs:
            # --- Training Phase ---
            net.train()
            total_loss = 0.0
            total_accuracy = 0.0
            total_adjusted_accuracy = 0.0
            # total_probability = 0.0 # This wasn't used, removed

            optimizer.zero_grad() # Zero gradients at the start of the epoch

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
                # total_probability += predicted_probabilities.sum().item() # This wasn't used

                loss = preference_loss(predicted_probabilities, batch_true_pref)
                accuracy = calculate_accuracy(predicted_probabilities, batch_true_pref)
                adjusted_accuracy = calculate_adjusted_accuracy(
                    predicted_probabilities, batch_true_pref, batch_score1, batch_score2
                )

                total_loss += loss.item() * batch_true_pref.size(0)
                total_accuracy += accuracy * batch_true_pref.size(0)
                total_adjusted_accuracy += adjusted_accuracy * batch_true_pref.size(0)

                loss.backward() # Calculate gradients

            # --- Training Step ---
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) # Clip gradients after accumulating for the epoch
            optimizer.step() # Update weights
            scheduler.step() # Update learning rate
            # --- End Training Step ---


            # Calculate and store average training metrics for the epoch
            average_training_loss = total_loss / train_size
            average_training_accuracy = total_accuracy / train_size
            average_adjusted_training_accuracy = total_adjusted_accuracy / train_size
            training_losses.append(average_training_loss)
            training_accuracies.append(average_training_accuracy)
            adjusted_training_accuracies.append(average_adjusted_training_accuracy)


            # --- Validation Phase (every validation_frequency epochs) ---
            if epoch % validation_frequency == 0:
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

                        total_validation_loss += validation_loss.item() * validation_true_pref.size(0) # Use .item()
                        total_validation_accuracy += validation_accuracy * validation_true_pref.size(0)
                        total_adjusted_validation_accuracy += adjusted_validation_accuracy * validation_true_pref.size(0)

                average_validation_loss = total_validation_loss / val_size
                average_validation_accuracy = total_validation_accuracy / val_size
                average_adjusted_validation_accuracy = (
                    total_adjusted_validation_accuracy / val_size
                )

                # Append validation metrics only when calculated
                validation_losses.append(average_validation_loss)
                validation_accuracies.append(average_validation_accuracy)
                adjusted_validation_accuracies.append(average_adjusted_validation_accuracy)

                wandb.log(
                    {
                        # Log training loss from the *current* epoch
                        "Train Loss": average_training_loss,
                        "Validation Loss": average_validation_loss,
                        "Train Accuracy": average_training_accuracy,
                        "Validation Accuracy": average_validation_accuracy,
                        "Adjusted Train Accuracy": average_adjusted_training_accuracy,
                        "Adjusted Validation Accuracy": average_adjusted_validation_accuracy,
                    },
                    step=epoch,
                )

                print(
                    f"Epoch {epoch}/{epochs}, Train Loss: {average_training_loss:.4f}, Val Loss: {average_validation_loss:.4f}, Train Acc: {average_training_accuracy:.4f} (adjusted: {average_adjusted_training_accuracy:.4f}), Val Acc: {average_validation_accuracy:.4f} (adjusted: {average_adjusted_validation_accuracy:.4f})"
                )

                # --- Early Stopping Check ---
                if average_validation_loss < best_loss:
                    best_loss = average_validation_loss
                    epochs_no_improve = 0 # Reset counter
                    # Store the state dict of the best model found so far
                    best_model_state = net.state_dict()
                    print(f"Epoch {epoch}: Validation loss improved to {best_loss:.4f}. Best model state updated.")
                else:
                    # Increment counter only when validation is checked and no improvement is found
                    epochs_no_improve += 1
                    print(f"Epoch {epoch}: Validation loss ({average_validation_loss:.4f}) did not improve from best ({best_loss:.4f}). Checks without improvement: {epochs_no_improve}/{patience}")

                if epochs_no_improve >= patience: # Check against hardcoded patience
                    print(f"Early stopping triggered after {epoch} epochs ({epochs_no_improve} checks without improvement).")
                    break # Exit the training loop
                # --- End Early Stopping Check ---

            # --- End Validation Block ---

            epoch += 1
    except Exception as e:
        # Save the *best* state found before exception, if available
        save_path = base_model_path + f"_EXCEPTION_{n_pairs}_pairs_{rules.NUMBER_OF_RULES}_rules.pth"
        if best_model_state:
             print("Saving best model state found before exception.")
             torch.save(best_model_state, save_path)
        else:
             print("Saving current model state before exception (no best state recorded).")
             torch.save(net.state_dict(), save_path)
        print(f"EXCEPTION CAUGHT during model training at epoch {epoch}: {e}")
        raise e # Re-raise the exception

    # --- Save the best model after training completes (or early stopping) ---
    try:
        n_pairs_magnitude = 10 ** round(math.log10(n_pairs)) if n_pairs > 0 else 0
    except ValueError: # Handle log10(0)
        n_pairs_magnitude = 0

    if best_model_state:
        final_save_path = base_model_path + f"_{n_pairs_magnitude}_pairs_{rules.NUMBER_OF_RULES}_rules.pth"
        torch.save(best_model_state, final_save_path)
        print(f"Best model state saved to {final_save_path} with validation loss: {best_loss:.4f}")
    else:
        # If no validation improvement ever happened (e.g., epochs < validation_frequency), save the last state.
        final_save_path = base_model_path + f"_{n_pairs_magnitude}_pairs_{rules.NUMBER_OF_RULES}_rules_LAST.pth"
        torch.save(net.state_dict(), final_save_path)
        print(f"No validation improvement recorded. Saving last model state to {final_save_path}")
    # --- End Save Best Model ---


    try:
        global figure_path
        # Ensure lists are suitable for plotting
        plot_train_losses = [l.item() if isinstance(l, torch.Tensor) else l for l in training_losses]
        plot_val_losses = [l.item() if isinstance(l, torch.Tensor) else l for l in validation_losses]
        plot_train_acc = [l.item() if isinstance(l, torch.Tensor) else l for l in training_accuracies]
        plot_val_acc = [l.item() if isinstance(l, torch.Tensor) else l for l in validation_accuracies]
        plot_adj_train_acc = [l.item() if isinstance(l, torch.Tensor) else l for l in adjusted_training_accuracies]
        plot_adj_val_acc = [l.item() if isinstance(l, torch.Tensor) else l for l in adjusted_validation_accuracies]

        # Create x-axis values for validation plots (sparse)
        validation_epochs = list(range(0, epoch, validation_frequency)) # Epochs where validation was run
        if epoch % validation_frequency == 0 and epoch > 0: # Include last epoch if validation ran
             pass # Already included by range
        elif epoch > 0 and len(validation_losses) > len(validation_epochs): # Handle early stopping case
             validation_epochs.append(epoch-1) # Add the epoch where it stopped

        plt.figure()
        plt.plot(plot_train_losses, label="Train Loss (per epoch)")
        # Plot validation loss if data exists
        if validation_epochs and plot_val_losses:
             # Ensure lengths match for plotting, potentially adjusting validation_epochs if needed due to early stop timing
             plot_epoch_count = min(len(validation_epochs), len(plot_val_losses))
             plt.plot(validation_epochs[:plot_epoch_count], plot_val_losses[:plot_epoch_count], label="Validation Loss (per check)", marker='o', linestyle='--')
        elif plot_val_losses:
             print(f"Could not plot validation loss: Mismatch between validation epochs ({len(validation_epochs)}) and recorded losses ({len(plot_val_losses)}).")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{figure_path}loss.png", dpi=600)
        plt.close()

        plt.figure()
        plt.plot(plot_train_acc, label="Train Accuracy (per epoch)")
        plt.plot(plot_adj_train_acc, label="Adjusted Training Accuracy (per epoch)")
        # Plot validation accuracy if data exists
        if validation_epochs and plot_val_acc:
             plot_epoch_count_acc = min(len(validation_epochs), len(plot_val_acc))
             plt.plot(validation_epochs[:plot_epoch_count_acc], plot_val_acc[:plot_epoch_count_acc], label="Validation Accuracy (per check)", marker='o', linestyle='--')
             # Plot adjusted validation accuracy if data exists and lengths match
             plot_epoch_count_adj_acc = min(len(validation_epochs), len(plot_adj_val_acc))
             if plot_adj_val_acc and plot_epoch_count_adj_acc > 0:
                 plt.plot(validation_epochs[:plot_epoch_count_adj_acc], plot_adj_val_acc[:plot_epoch_count_adj_acc], label="Adjusted Validation Accuracy (per check)", marker='x', linestyle=':')
             elif plot_adj_val_acc:
                 print(f"Could not plot adjusted validation accuracy: Mismatch between validation epochs ({len(validation_epochs)}) and recorded adjusted accuracies ({len(plot_adj_val_acc)}).")
        elif plot_val_acc:
             print(f"Could not plot validation accuracy: Mismatch between validation epochs ({len(validation_epochs)}) and recorded accuracies ({len(plot_val_acc)}).")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{figure_path}accuracy.png", dpi=600)
        plt.close()
    except Exception as plot_err:
        print(f"Issue when plotting training/validation metrics: {plot_err}")

    # Return the best validation loss achieved
    training_output = best_loss # best_loss holds the best validation loss
    if return_stat == "acc":
        # Accuracies are from the *last calculated validation epoch*
        training_output = {
            # Use last training accuracy available
            "final_training_acc": training_accuracies[-1] if training_accuracies else 0,
            "final_validation_acc": validation_accuracies[-1] if validation_accuracies else 0,
            "final_adjusted_training_acc": adjusted_training_accuracies[-1] if adjusted_training_accuracies else 0,
            "final_adjusted_validation_acc": adjusted_validation_accuracies[-1] if adjusted_validation_accuracies else 0,
            "best_validation_loss": best_loss
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
    save_at_end=True, # This flag is no longer needed, saving handled internally
):
    torch.cuda.empty_cache()
    training_output_stat = None
    input_size = INPUT_SIZE
    # ensemble_path setup moved inside train_ensemble

    if figure_folder_name:
        global figure_path
        figure_path = figure_folder_name
        if figure_path[-1] != "/":
            figure_path += "/"
        os.makedirs(figure_path, exist_ok=True)
    
    model_id = "".join([str(rule) for rule in rules.RULES_INCLUDED])
    print("MODEL ID:", model_id)


    # OPTUNA (No changes needed regarding patience)
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
        print(f"Value: {best_trial.value}") # This is the best validation loss found by Optuna
        print(f"Params: {best_trial.params}")

        with open("best_params.yaml", "w") as f:
            yaml.dump(best_trial.params, f)
            print("Best hyperparameters saved to best_params.yaml.")

        # Note: The actual saving of the *best model* from the best Optuna trial
        # now happens *inside* train_model/train_ensemble during that trial's execution.
        # We don't need to reload and resave here. Optuna's `best_trial.value`
        # already represents the best validation loss achieved according to the objective.

        # Cleanup Optuna temporary model files if they were created (depends on objective implementation)
        # Example cleanup (adjust pattern if needed):
        # for file in glob.glob(models_path + "best_model_trial_*.pth"):
        #     os.remove(file)
        # for file in glob.glob(models_path + "best_ensemble_trial_*/*"):
        #     try:
        #         os.remove(file)
        #     except IsADirectoryError:
        #         pass # Ignore directories if pattern matches them
        # for dir_path in glob.glob(models_path + "best_ensemble_trial_*"):
        #      if os.path.isdir(dir_path):
        #          try:
        #              os.rmdir(dir_path) # Remove empty directory
        #          except OSError:
        #              print(f"Could not remove directory {dir_path}, might not be empty.")


    # NO OPTUNA
    else:
        print("RUNNING WITHOUT OPTUNA:")
        with open(parameters_path, "r") as file:
            data = yaml.safe_load(file)
            hidden_size = data["hidden_size"]
            learning_rate = data["learning_rate"]
            weight_decay = data["weight_decay"]
            dropout_prob = data["dropout_prob"]
            swaps = data.get("swaps", 3) # Use .get for optional parameters
            batch_size = data["batch_size"]
            # No need to read 'patience' from YAML anymore

            if use_ensemble:
                ensemble = Ensemble(
                    input_size, ENSEMBLE_SIZE, None, hidden_size, dropout_prob
                ).to(device)
                for param in ensemble.parameters():
                    if len(param.shape) > 1:
                        nn.init.xavier_uniform_(
                            param, gain=nn.init.calculate_gain("relu")
                        )
                optimizer = torch.optim.Adam(
                    ensemble.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                # Call train_ensemble without patience argument
                training_output_stat = train_ensemble(
                    ensemble=ensemble,
                    epochs=epochs,
                    swaps=swaps,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    trajectory_path=trajectories_file_path,
                    return_stat=return_stat,
                    preload=True, # Assuming preload=True for non-Optuna runs
                    # No patience argument needed
                )
                # Saving logic is now inside train_ensemble
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
                # Call train_model without patience argument
                training_output_stat = train_model_without_dataloader(
                    file_path=trajectories_file_path,
                    net=net,
                    epochs=epochs,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    base_model_path=models_path + f"model_{model_id}_{epochs}_epochs", # Base path for saving
                    return_stat=return_stat,
                    preload=True, # Assuming preload=True for non-Optuna runs
                    # No patience argument needed
                )

                # Saving logic is now inside train_model
        return training_output_stat


def objective(trial):
    # This function defines a single Optuna trial for the non-ensemble case.
    # It suggests hyperparameters, creates a model, trains it using train_model,
    # and returns the best validation loss achieved during that training run.
    input_size = INPUT_SIZE
    hidden_size = trial.suggest_int("hidden_size", 64, 1024)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True) # Log scale often better for LR
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # Log scale
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.7) # Adjusted range slightly
    batch_size = trial.suggest_int("batch_size", 2048, 16384) # Adjusted range
    # swaps = trial.suggest_int("swaps", 0, 5) # Swaps not relevant for single model

    net = TrajectoryRewardNet(input_size, hidden_size, dropout_prob).to(device)
    for param in net.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # train_model now handles early stopping and returns the best validation loss
    # It also saves the best model internally during the trial.
    best_validation_loss = train_model(
        file_path=study.user_attrs["file_path"],
        net=net,
        epochs=study.user_attrs["epochs"],
        optimizer=optimizer,
        batch_size=batch_size,
        model_path=models_path + f"model_trial_{trial.number}", # Unique path for this trial's best model
        preload=True, # Assuming preload=True for Optuna runs
    )

    # Optuna needs a value to minimize. Return the best validation loss.
    # No need to explicitly save the model here anymore, train_model does it.
    # Pruning can still be added if desired, based on intermediate validation losses reported by train_model.
    # trial.report(best_validation_loss, step=?) # Need to modify train_model to report intermediate steps for pruning

    return best_validation_loss # Return the metric Optuna should minimize


def ensemble_objective(trial):
    # This function defines a single Optuna trial for the ensemble case.
    input_size = INPUT_SIZE
    hidden_size = trial.suggest_int("hidden_size", 64, 1024)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.7)
    batch_size = trial.suggest_int("batch_size", 2048, 16384)
    swaps = trial.suggest_int("swaps", 0, 10) # Swaps are relevant for ensemble

    # Use a unique path for this trial's ensemble models
    trial_ensemble_base_path = models_path + f"ensemble_trial_{trial.number}/"
    # Note: train_ensemble will create the final path including epoch/pair counts based on this

    ensemble = Ensemble(
        input_size=input_size,
        num_models=ENSEMBLE_SIZE,
        models_list=None, # Create new models for the trial
        hidden_size=hidden_size,
        dropout_prob=dropout_prob
    ).to(device)

    for param in ensemble.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))
    optimizer = torch.optim.Adam(
        ensemble.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # train_ensemble now handles early stopping and returns the best avg validation loss
    # It also saves the best ensemble models internally during the trial.
    best_avg_validation_loss = train_ensemble(
        ensemble=ensemble,
        epochs=study.user_attrs["epochs"],
        swaps=swaps,
        optimizer=optimizer,
        batch_size=batch_size,
        trajectory_path=study.user_attrs["file_path"],
        preload=True, # Assuming preload=True for Optuna runs
        # model_path is handled internally by train_ensemble now
    )

    # Optuna needs a value to minimize. Return the best average validation loss.
    # No need to explicitly save the models here anymore.
    # trial.report(best_avg_validation_loss, step=?) # For pruning

    return best_avg_validation_loss # Return the metric Optuna should minimize


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
    # parse.add_argument("-l", "--labels", type=int, help="Number of labels created") # Labels arg seems unused
    parse.add_argument(
        "--ensemble", action="store_true", help="Train an ensemble of predictors" # Default ENSEMBLE_SIZE=3
    )
    parse.add_argument(
        "-p",
        "--parameters",
        type=str,
        help="Directory to hyperparameter yaml file (skips Optuna if provided)",
    )
    args = parse.parse_args()
    if args.database:
        data_path = args.database
    else:
        # Provide a default or raise an error if database is required
        # data_path = "trajectories/database_350.pkl"
        raise ValueError("Trajectory database file path (-d/--database) is required.")


    if args.epochs:
        epochs = args.epochs
    else:
        # Provide a default or raise an error if epochs are required
        # epochs = 50
        raise ValueError("Number of epochs (-e/--epochs) is required.")

    # The train_reward_function now handles Optuna vs non-Optuna runs
    # and calls the appropriate training function (train_model or train_ensemble)
    # which internally handle early stopping and saving the best model(s).
    train_reward_function(
        trajectories_file_path=data_path,
        epochs=epochs,
        parameters_path=args.parameters,
        use_ensemble=args.ensemble
        # return_stat can be added if needed, e.g., return_stat="acc"
    )

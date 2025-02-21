#!/bin/bash

# Function to check if SSH to GitHub works
check_ssh() {
    ssh -T git@github.com &>/dev/null
    return $?
}

# Check SSH authentication with GitHub
if check_ssh; then
    echo "SSH authentication with GitHub is already working."
else
    echo "SSH authentication failed. Attempting to authenticate with GitHub CLI."
    gh auth login
fi

# Check GitHub authentication status
gh auth status

# Set the remote URL for Git to the specified repository
git remote set-url origin https://github.com/llama887/ai-car-preference-learning.git

# Ensure W&B is installed and log in
if ! command -v wandb &>/dev/null; then
    echo "wandb not found, installing now..."
    pip install wandb
fi

# Log into Weights & Biases
if ! wandb whoami &>/dev/null; then
    echo "W&B not logged in, attempting login..."
    echo "find api key at https://wandb.ai/authorize"
    wandb login
else
    echo "W&B already authenticated."
fi

#!/bin/bash

# Define the environment path and requirements file
ENV_PATH="environments/pref_learning"
REQUIREMENTS_FILE="environments/linux_requirements.txt"

# Check if the virtual environment exists, if not, create it
if [ ! -d "$ENV_PATH" ]; then
    echo "Virtual environment not found. Creating it..."
    python3 -m venv "$ENV_PATH"
    source "$ENV_PATH/bin/activate"
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "Environment setup complete."
else
    # Activate the virtual environment if it already exists
    echo "Activating the virtual environment..."
    source "$ENV_PATH/bin/activate"
fi

# Default array of trajectories if none are provided as input
TRAJECTORIES=(1000000 1000000 10000)

# Fixed parameters
EPOCHS=1000
GENERATIONS=100
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="main.py"

# Check if any arguments are passed for trajectories
if [ "$#" -gt 0 ]; then
    TRAJECTORIES=("$@")
fi

# Loop over each trajectory value
for TRAJ in "${TRAJECTORIES[@]}"; do
    # Remove the directories to prepare for the next run
    rm -rf figures trajectories

    echo "Running with ${TRAJ} trajectories..."

    # Run the main.py script
    python "$MAIN_SCRIPT" -e "$EPOCHS" -t "$TRAJ" -g "$GENERATIONS" -p "$PARAM_FILE" --headless

    # Check if the directories exist and zip them
    if [ -d "figures" ]; then
        zip -r "figures_${TRAJ}.zip" figures
    else
        echo "Warning: figures directory not found for ${TRAJ} trajectories."
    fi

    if [ -d "trajectories" ]; then
        zip -r "trajectories_${TRAJ}.zip" trajectories
    else
        echo "Warning: trajectories directory not found for ${TRAJ} trajectories."
    fi

    echo "Completed run with ${TRAJ} trajectories."
done

echo "All runs completed."

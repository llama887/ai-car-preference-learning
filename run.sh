#!/bin/bash

# Activate the virtual environment
source environments/pref_learning/bin/activate

# Array of trajectories to iterate over
TRAJECTORIES=(10000 100000 1000000 10000000)

# Fixed parameters
EPOCHS=1000
GENERATIONS=50
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="main.py"

# Loop over each trajectory value
for TRAJ in "${TRAJECTORIES[@]}"; do
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

    # Remove the directories to prepare for the next run
    rm -rf figures trajectories

    echo "Completed run with ${TRAJ} trajectories."
done

echo "All runs completed."

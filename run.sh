#!/bin/bash

# Activate the virtual environment
source environments/pref_learning/bin/activate

# Array of trajectories to iterate over
TRAJECTORIES=(1000000 100000 10000)

# Fixed parameters
EPOCHS=1000
GENERATIONS=50
NUM_RULES=2
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="main.py"


# Loop over each trajectory value
for TRAJ in "${TRAJECTORIES[@]}"; do
    # Remove the directories to prepare for the next run
    rm -rf figures trajectories

    echo "Running with ${TRAJ} trajectories... and learning ${NUM_RULES} rules."

    # Run the main.py script
    python "$MAIN_SCRIPT" -e "$EPOCHS" -t "$TRAJ" -g "$GENERATIONS" -p "$PARAM_FILE" -c "$NUM_RULES" --headless

    # Check if the directories exist and zip them
    if [ -d "figures" ]; then
        zip -r "figures_${TRAJ}_pairs_${NUM_RULES}_rules.zip" figures
    else
        echo "Warning: figures directory not found for ${TRAJ} trajectories and ${NUM_RULES} rules."
    fi

    if [ -d "trajectories" ]; then
        zip -r "trajectories_${TRAJ}_pairs_${NUM_RULES}_rules.zip" trajectories
    else
        echo "Warning: trajectories directory not found for ${TRAJ} trajectories and ${NUM_RULES} rules."
    fi

    echo "Completed run with ${TRAJ} trajectories."
done

echo "All runs completed."

#!/bin/bash

# Default array of trajectories if none are provided as input
TRAJECTORIES=(1000000 1000000 10000)

# Function to display usage
usage() {
    echo "Usage: $0 -r <rules>"
    exit 1
}

# Parse arguments
rules=""

while getopts "r:" opt; do
    case "$opt" in
        r)
            rules=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

# Validate the rules input
if ! [[ $rules =~ ^[0-9]+$ ]]; then
    echo "Error: The -r option must be an integer."
    usage
fi

# Create the list of distribution values
distribution=$(printf -- "-d '1/%d' " $(seq 1 $((rules+1)) | sed "s/.*/$((rules+1))/"))
echo $distribution

# Fixed parameters
EPOCHS=200
GENERATIONS=100
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="main.py"

# Remove any existing zip files for figures and trajectories to avoid conflicts
rm -f figures_*.zip trajectories_*.zip

# Create the zips directory if it doesn't exist
mkdir zips

# Loop over each trajectory value
for TRAJ in "${TRAJECTORIES[@]}"; do
    # Remove the directories to prepare for the next run
    rm -rf figures trajectories

    echo "Running with ${TRAJ} trajectories..."

    # Run the main.py script
    python "$MAIN_SCRIPT" -e "$EPOCHS" -t "$TRAJ" -g "$GENERATIONS" -p "$PARAM_FILE" -c "$rules" "$distribution" --headless
    echo "python $MAIN_SCRIPT -e $EPOCHS -t $TRAJ -g $GENERATIONS -p $PARAM_FILE -c $rules $distribution --headless"
    # Check if the directories exist and zip them
    if [ -d "figures" ]; then
        zip -r "zips/figures_t${TRAJ}_r${rules}.zip" figures
    else
        echo "Warning: figures directory not found for ${TRAJ} trajectories."
    fi

    if [ -d "trajectories" ]; then
        zip -r "zips/trajectories_t${TRAJ}_r${rules}.zip" trajectories
    else
        echo "Warning: trajectories directory not found for ${TRAJ} trajectories."
    fi

    echo "Completed run with ${TRAJ} trajectories."
done

echo "All runs completed."

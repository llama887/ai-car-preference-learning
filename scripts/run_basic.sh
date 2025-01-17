#!/bin/bash

# Default array of trajectories if none are provided as input
TRAJECTORIES=(1000000 100000 10000)

# Function to display usage
usage() {
    echo "Usage: $0 -r <rules> [-p]"
    echo "  -r <rules>   Specify the rules value (integer)"
    echo "  -p           Run in parallel"
    exit 1
}

# Parse arguments
rules=""
parallel=false

while getopts "r:p" opt; do
    case "$opt" in
        r)
            rules=$OPTARG
            ;;
        p)
            parallel=true
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
distribution=$(printf -- "-d \"1/%d\" " $(seq 1 $((rules+1)) | sed "s/.*/$((rules+1))/"))

# Fixed parameters
EPOCHS=400
GENERATIONS=100
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="main.py"

# Remove any existing zip files for figures and trajectories to avoid conflicts
rm -f figures_*.zip trajectories_*.zip

# Create the zips directory if it doesn't exist
mkdir -p zips

# Function to run a single instance of main.py
run_instance() {
    TRAJ=$1
    FIGURE_DIR="figures_t$TRAJ"
    TRAJECTORY_DIR="trajectories_t$TRAJ"

    # Remove the directories to prepare for the next run
    rm -rf figures* trajectories trajectories_t*

    echo "Running with ${TRAJ} trajectories..."

    # Run the main.py script
    cmd="python $MAIN_SCRIPT -e $EPOCHS -t $TRAJ -g $GENERATIONS -p $PARAM_FILE -c $rules --figure $FIGURE_DIR --trajectory $TRAJECTORY_DIR $distribution --headless"
    echo "Executing: $cmd"
    eval $cmd

    # Check if the directories exist and zip them
    if [ -d "$FIGURE_DIR" ]; then
        zip -r "zips/${FIGURE_DIR}_r${rules}.zip" $FIGURE_DIR
    else
        echo "Warning: $FIGURE_DIR not found for ${TRAJ} trajectories."
    fi

    if [ -d "$TRAJECTORY_DIR" ]; then
        zip -r "zips/${TRAJECTORY_DIR}_r${rules}.zip" $TRAJECTORY_DIR
    else
        echo "Warning: $TRAJECTORY_DIR not found for ${TRAJ} trajectories."
    fi

    echo "Completed run with ${TRAJ} trajectories."
}

# Export the function and variables so they are available to parallel processes
export -f run_instance
export MAIN_SCRIPT EPOCHS GENERATIONS PARAM_FILE rules distribution

# Run instances either in parallel or sequentially
if $parallel; then
    parallel --ungroup -j 3 run_instance ::: "${TRAJECTORIES[@]}"

else
    for TRAJ in "${TRAJECTORIES[@]}"; do
        run_instance $TRAJ
    done
fi

echo "All runs completed."
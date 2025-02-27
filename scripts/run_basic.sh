#!/bin/bash

# Default array of trajectories if none are provided as input
TRAJECTORIES=(1000000 100000 10000 1000)

# Function to display usage
usage() {
    echo "Usage: $0 -r <rules> [-p] [-e] [-h] [-s]"
    echo "  -r <rules>     Specify the rules value (integer)"
    echo "  -p             Run in parallel"
    echo "  -e             Enable ensembling"
    echo "  -h             Enable heatmap"
    echo "  -s             Enable subsampling"
    exit 1
}

# Parse arguments
rules=""
parallel=false
ensembling=false
heatmap=false
subsample=false

while getopts ":r:pehs" opt; do
    case "$opt" in
        r)
            rules=$OPTARG
            ;;
        p)
            parallel=true
            ;;
        e)
            ensembling=true
            ;;
        h)
            heatmap=true
            ;;
        s)
            subsample=true
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
distribution=$(printf -- "-d \"1/%d\" " $(seq 1 $rules | sed "s/.*/$((2 * rules))/"); printf -- "-d \"1/2\"")

# Fixed parameters
EPOCHS=3000
GENERATIONS=200
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="main.py"

# Create the zips directory if it doesn't exist
mkdir -p zips_baseline
mkdir -p zips_ensembling
mkdir -p zips_subsample
mkdir -p logs


# Remove previous results
rm -rf figures* trajectories trajectories_t*

# Function to run a single instance of main.py
run_instance() {
    TRAJ=$1
    FIGURE_DIR="figures_t$TRAJ"
    TRAJECTORY_DIR="trajectories_t$TRAJ"
    ZIP_DIR="zips"

    echo "Running with ${TRAJ} trajectories..."

    # Define the command
    cmd="stdbuf -oL python -u $MAIN_SCRIPT -e $EPOCHS -t $TRAJ -g $GENERATIONS -p $PARAM_FILE -c $rules --figure $FIGURE_DIR --trajectory $TRAJECTORY_DIR $distribution --headless --skip-plots --save-at-end --skip-retrain"

    if $heatmap; then
        cmd+=" --heatmap"
    fi

    if $subsample; then
        cmd+=" -md subsampled_gargantuar_1_length_${rules}_rules.pkl"
    fi

    ZIP_SUFFIX=""
    if $ensembling; then
        cmd+=" --ensemble"
        ZIP_SUFFIX+="ensembling"
        ZIP_DIR+="ensembling"
    elif $subsample; then
        ZIP_SUFFIX+="_subsample"
        ZIP_DIR+="_subsample"
    else
        ZIP_DIR+="_baseline"
    fi

    LOG_FILE="logs/log_${TRAJ}_t_${rules}_r${ZIP_SUFFIX}.log"

    # Retry logic
    MAX_RETRIES=5
    RETRY_DELAY=300  # Initial wait time (seconds)
    attempt=1

    while [ $attempt -le $MAX_RETRIES ]; do
        echo "Attempt $attempt: Executing: $cmd 2>&1 | tee $LOG_FILE"
        eval $cmd 2>&1 | tee $LOG_FILE || $cmd 2>&1 | tee $LOG_FILE
        EXIT_CODE=${PIPESTATUS[0]}

        # Check for OOM conditions
        if grep -qi "out of memory\|killed process" $LOG_FILE || [ $EXIT_CODE -eq 137 ]; then
            echo "Detected OOM error! Retrying in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
            RETRY_DELAY=$((RETRY_DELAY * 2))  # Exponential backoff
            attempt=$((attempt + 1))
        else
            break  # Exit loop if successful
        fi
    done

    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "Run failed after $MAX_RETRIES attempts. Skipping..."
        return 1
    fi

    # Zip results
    if [ -d "$FIGURE_DIR" ]; then
        zip -r "${ZIP_DIR}/${FIGURE_DIR}_r${rules}${ZIP_SUFFIX}.zip" $FIGURE_DIR
    else
        echo "Warning: $FIGURE_DIR not found for ${TRAJ} trajectories."
    fi

    if [ -d "$TRAJECTORY_DIR" ]; then
        zip -r "${ZIP_DIR}/${TRAJECTORY_DIR}_r${rules}${ZIP_SUFFIX}.zip" $TRAJECTORY_DIR
    else
        echo "Warning: $TRAJECTORY_DIR not found for ${TRAJ} trajectories."
    fi

    echo "Completed run with ${TRAJ} trajectories."

}


# Export the function and variables so they are available to parallel processes
export -f run_instance
export MAIN_SCRIPT EPOCHS GENERATIONS PARAM_FILE rules distribution segments ensembling heatmap parallel subsample

# Run instances either in parallel or sequentially
if $parallel; then
    parallel --ungroup -j 4 run_instance ::: "${TRAJECTORIES[@]}"
else
    for TRAJ in "${TRAJECTORIES[@]}"; do
        run_instance $TRAJ
    done
fi

echo "All runs completed."

#!/bin/bash

# Default array of trajectories if none are provided as input
TRAJECTORIES=1000000
rules=3

# Function to display usage
usage() {
    echo "Usage: $0 -s <max_segment_lengths> [-p]"
    echo "  -s <max_segment_lengths>     Specify the rules value (integer)"
    echo "  -p             Run in parallel"
    exit 1
}

# Parse arguments
max_segment_lengths=""
parallel=false

while getopts "s:p" opt; do
    case "$opt" in
        s)
            max_segment_lengths=$OPTARG
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
if ! [[ $max_segment_lengths =~ ^[0-9]+$ ]]; then
    echo "Error: The -s option must be an integer."
    usage
fi

# Create the list of distribution values
distribution=$(printf -- "-d \"1/%d\" " $(seq 1 $rules | sed "s/.*/$((2 * rules))/"); printf -- "-d \"1/2\"")

# Fixed parameters
EPOCHS=100
GENERATIONS=200
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="main.py"

# Remove any existing zip files for figures and trajectories to avoid conflicts
mv zips zips_last

# Create the zips directory if it doesn't exist
mkdir -p zips

# Function to run a single instance of main.py
run_instance() {
    SEGMENT_LENGTH=$1
    FIGURE_DIR="figures_s$SEGMENT_LENGTH"
    TRAJECTORY_DIR="trajectories_s$SEGMENT_LENGTH"

    # Remove the directories to prepare for the next run
    rm -rf figures* trajectories trajectories_t*

     echo "Running with segment length ${SEGMENT_LENGTH}..."

    # Run the main.py script
    cmd="python $MAIN_SCRIPT -e $EPOCHS -t $(expr $TRAJECTORIES / $SEGMENT_LENGTH)  -g $GENERATIONS -p $PARAM_FILE -c $rules --figure $FIGURE_DIR --trajectory $TRAJECTORY_DIR $distribution --headless --skip-plots -s $SEGMENT_LENGTH"
    
    echo "Executing: $cmd"
    eval $cmd

    # Check if the directories exist and zip them
    if [ -d "$FIGURE_DIR" ]; then
        zip -r "zips/${FIGURE_DIR}_r${rules}.zip" $FIGURE_DIR
    else
        echo "Warning: $FIGURE_DIR not found for ${SEGMENT_LENGTH} segment length."
    fi

    if [ -d "$TRAJECTORY_DIR" ]; then
        zip -r "zips/${TRAJECTORY_DIR}_r${rules}.zip" $TRAJECTORY_DIR
    else
        echo "Warning: $TRAJECTORY_DIR not found for ${SEGMENT_LENGTH} segment length."
    fi

    echo "Completed run with ${SEGMENT_LENGTH} segment length."
}

# Export the function and variables so they are available to parallel processes
export -f run_instance
export MAIN_SCRIPT EPOCHS GENERATIONS PARAM_FILE rules distribution TRAJECTORIES

# Run instances for all segment lengths up to the specified value
if $parallel; then
    parallel --ungroup -j 3 run_instance ::: $(seq 1 $max_segment_length)
else
    for SEGMENT_LENGTH in $(seq 1 $max_segment_length); do
        run_instance $SEGMENT_LENGTH
    done
fi

echo "All runs completed."
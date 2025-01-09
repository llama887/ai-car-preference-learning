#!/bin/bash

# Fixed parameters
EPOCHS=400
TRAJECTORIES=1000000
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="partial_rewards.py"

# Parse arguments
resolution=""

while getopts "r:" opt; do
    case "$opt" in
        r)
            resolution=$OPTARG
            ;;
        *)
            usage
            ;;
    esac
done

# Validate the rules input
if ! [[ $resolution =~ ^[0-9]+$ ]]; then
    echo "Error: The -r option must be an integer."
    usage
fi

# Run the script
cmd="python $MAIN_SCRIPT -e $EPOCHS -t $TRAJ -r $resolution -p $PARAM_FILE --headless"
echo "Executing: $cmd"
eval $cmd



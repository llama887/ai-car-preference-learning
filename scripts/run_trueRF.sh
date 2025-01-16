#!/bin/bash

RULES=(1 2 3)

# Function to display usage
usage() {
    echo "Usage: $0 [-p]"
    echo "  -p           Run in parallel"
    exit 1
}

# Parse arguments
parallel=false

while getopts "p" opt; do
    case "$opt" in
        p)
            parallel=true
            ;;
        *)
            usage
            ;;
    esac
done

# Fixed parameters
GENERATIONS=100
MAIN_SCRIPT="generate_trueRF.py"

# Function to run a single instance of main.py
run_instance() {
    RULE=$1
    # Run the main.py script
    cmd="python $MAIN_SCRIPT -g $GENERATIONS -c $RULE --headless"
    echo "Executing: $cmd"
    eval $cmd

    echo "Completed run with ${RULE} rules."
}

# Export the function and variables so they are available to parallel processes
export -f run_instance
export MAIN_SCRIPT EPOCHS GENERATIONS PARAM_FILE rules distribution

# Run instances either in parallel or sequentially
if $parallel; then
    parallel --ungroup -j 3 run_instance ::: "${RULES[@]}"

else
    for RULE in "${RULES[@]}"; do
        run_instance $RULE
    done
fi

echo "All runs completed."
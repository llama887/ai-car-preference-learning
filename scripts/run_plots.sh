#!/bin/bash
echo "Plotting performance plots"

PLOT_SCRIPT="performance_plots.py"


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


cmd="python $PLOT_SCRIPT -c $rules"
echo "Executing: $cmd"
eval $cmd
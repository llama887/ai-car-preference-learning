#!/bin/bash

trajectories=""
rules=""
segment_length=""
number_of_processes=""
paired=false

while getopts "t:s:n:p" flag; do
    case "${flag}" in
        t) trajectories=${OPTARG};;
        s) segment_length=${OPTARG};;
        n) number_of_processes=${OPTARG};;
        p) paired=${OPTARG};;
        *) echo "Usage: $0 -t trajectories -n number_of_processes -s segment_length -p paired_database " >&2
           exit 1 ;;
    esac
done

# Validate inputs
if [[ -z $trajectories || -z $number_of_processes ]]; then
    echo "All arguments (-t, -n) are required." >&2
    exit 1
fi

# Divide the trajectories by the number of processes
trajectories_per_process=$((trajectories / number_of_processes))

mv tmp tmp_old
# Create a temporary directory
mkdir -p tmp

# Run Python scripts in parallel
for ((i=0; i<number_of_processes; i++)); do
    cmd="stdbuf -oL python -u collect_data.py -t $trajectories_per_process  -db tmp/master_database_${i}.pkl --trajectory tmp/trajectory_${i}/ --headless"
    if [[ -n $segment_length ]]; then
        cmd="$cmd -s $segment_length"
    fi
    if $paired; then
        cmd+=" -p"
    fi
    echo "Executing: $cmd"
    eval $cmd | tee tmp/output_$i.log &
done

# Wait for all background processes to complete
wait

echo "All processes completed."

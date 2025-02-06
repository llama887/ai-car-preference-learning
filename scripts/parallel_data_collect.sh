#!/bin/bash

trajectories=""
rules=""
segment_length=""
number_of_processes=""
partial_rewards=false

while getopts "t:r:s:n:g" flag; do
    case "${flag}" in
        t) trajectories=${OPTARG};;
        r) rules=${OPTARG};;
        s) segment_length=${OPTARG};;
        n) number_of_processes=${OPTARG};;
        *) echo "Usage: $0 -t trajectories -r rules -n number_of_processes -s segment_length" >&2
           exit 1 ;;
    esac
done

# Validate inputs
if [[ -z $trajectories || -z $rules || -z $number_of_processes ]]; then
    echo "All arguments (-t, -r, -n) are required." >&2
    exit 1
fi

# Divide the trajectories by the number of processes
trajectories_per_process=$((trajectories / number_of_processes))

# Create the list of distribution values
distribution=$(printf -- "-d \"1/%d\" " $(seq 1 $rules | sed "s/.*/$((2 * rules))/"); printf -- "-d \"1/2\"")
echo "Distribution: $distribution"


mv tmp tmp_old
# Create a temporary directory
mkdir -p tmp

# Run Python scripts in parallel
for ((i=0; i<number_of_processes; i++)); do
    cmd="stdbuf -oL python -u collect_data.py -t $trajectories_per_process $distribution -db tmp/master_database_${i}.pkl --trajectory tmp/trajectory_${i}/ --headless"
    if [[ -n $segment_length ]]; then
        cmd="$cmd -s $segment_length"
    fi
    echo "Executing: $cmd"
    eval $cmd | tee tmp/output_$i.log &
done

# Wait for all background processes to complete
wait

echo "All processes completed."

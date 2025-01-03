#!/bin/bash

trajectories=""
rules=""
number_of_processes=""

while getopts "t:r:n:" flag; do
    case "${flag}" in
        t) trajectories=${OPTARG};;
        r) rules=${OPTARG};;
        n) number_of_processes=${OPTARG};;
        *) echo "Usage: $0 -t trajectories -r rules -n number_of_processes" >&2
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
distribution=$(printf -- "-d \"1/%d\" " $(seq 1 $((rules+1)) | sed "s/.*/$((rules+1))/"))
echo "Distribution: $distribution"
# Create a temporary directory
mkdir -p tmp

# Run Python scripts in parallel
for ((i=0; i<number_of_processes; i++)); do
    cmd="stdbuf -oL python collect_data.py -t $trajectories_per_process $distribution -db tmp/master_database_$i -tp tmp/trajectory_${i}_ --headless"
    echo "Executing: $cmd"
    eval $cmd | tee tmp/output_$i.log &
done

# Wait for all background processes to complete
wait

echo "All processes completed."

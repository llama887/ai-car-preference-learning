#!/bin/bash

trajectories=""
rules=""
segment_length=""
number_of_processes=""
paired=false
storage_dir="tmp"
MAX_PARALLEL=36

while getopts "t:r:s:n:p:d:" flag; do
    case "${flag}" in
        t) trajectories=${OPTARG};;
        r) rules=${OPTARG};;
        s) segment_length=${OPTARG};;
        n) number_of_processes=${OPTARG};;
        p) paired=true;;
        d) storage_dir=${OPTARG};;
        *) echo "Usage: $0 -t trajectories -r rules -n number_of_processes [-s segment_length] [-p] [-d storage_dir]" >&2
           exit 1 ;;
    esac
done

if [[ -z $trajectories || -z $rules || -z $number_of_processes ]]; then
    echo "All arguments (-t, -r, -n) are required." >&2
    exit 1
fi

# Get the task index from SLURM or fall back to loop logic
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    index=$SLURM_ARRAY_TASK_ID
    trajectories_per_process=$((trajectories / number_of_processes))
    distribution=$(printf -- "-d \"1/%d\" " $(seq 1 $rules); printf -- "-d \"1/2\"")

    mkdir -p "$storage_dir/trajectory_${index}"

    cmd="stdbuf -oL python -u collect_data.py -t $trajectories_per_process $distribution -db $storage_dir/master_database_${index}.pkl --trajectory $storage_dir/trajectory_${index}/ --headless"
    [[ -n $segment_length ]] && cmd="$cmd -s $segment_length"
    [[ $paired == true ]] && cmd="$cmd -p"

    echo "Executing (array mode): $cmd"
    eval "$cmd" | tee "$storage_dir/output_${index}.log"

else
    # Local loop mode (no SLURM array)
    trajectories_per_process=$((trajectories / number_of_processes))
    distribution=$(printf -- "-d \"1/%d\" " $(seq 1 $rules); printf -- "-d \"1/2\"")
    echo "Distribution: $distribution"

    [[ -d $storage_dir ]] && mv "$storage_dir" "${storage_dir}_old"
    mkdir -p "$storage_dir"

    job_counter=0

    for ((i=0; i<number_of_processes; i++)); do
        cmd="stdbuf -oL python -u collect_data.py -t $trajectories_per_process $distribution -db $storage_dir/master_database_${i}.pkl --trajectory $storage_dir/trajectory_${i}/ --headless"
        [[ -n $segment_length ]] && cmd="$cmd -s $segment_length"
        [[ $paired == true ]] && cmd="$cmd -p"
        echo "Executing: $cmd"
        eval "$cmd" | tee "$storage_dir/output_${i}.log" &
        ((job_counter++))
        if (( job_counter % MAX_PARALLEL == 0 )); then
            wait
        fi
    done

    wait
    echo "All processes completed."
fi


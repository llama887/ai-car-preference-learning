#!/bin/bash

# Fixed parameters
EPOCHS=200
TRAJ=500000
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="partial_rewards.py"

usage() {
    echo "Usage: $0 -r <resolution> -p <parallelism>"
    echo "  -r    Resolution value (must be a positive integer)."
    echo "  -p    Number of parallel executions (must be a positive integer)."
    exit 1
}

RESOLUTION=""
PARALLELISM=1  

while getopts "r:p:" opt; do
    case "$opt" in
        r) RESOLUTION=$OPTARG ;;
        p) PARALLELISM=$OPTARG ;;
        *) usage ;;
    esac
done

if ! [[ "$RESOLUTION" =~ ^[0-9]+$ ]] || [ "$RESOLUTION" -le 0 ]; then
    echo "Error: The -r option must be a positive integer."
    usage
fi

if ! [[ "$PARALLELISM" =~ ^[0-9]+$ ]] || [ "$PARALLELISM" -le 0 ]; then
    echo "Error: The -p option must be a positive integer."
    usage
fi

rm -rf trajectories_partial*

STEP=$(echo "scale=10; 1 / $RESOLUTION" | bc)
I=0
CURRENT_JOBS=0

for A in $(seq 0 $STEP 1); do
    for B in $(seq 0 $STEP $(echo "1 - $A" | bc)); do
        C=$(echo "1 - $A - $B" | bc)
        if (( $(echo "$C < 0" | bc) )); then
            C=0
        fi
        CMD="python $MAIN_SCRIPT -e $EPOCHS -t $TRAJ -r $RESOLUTION -p $PARAM_FILE -a $A -b $B -c $C -i $I --headless"
        echo "Executing: $CMD"
        eval $CMD &
        CURRENT_JOBS=$((CURRENT_JOBS + 1))
        I=$((I + 1))

        if (( CURRENT_JOBS >= PARALLELISM )); then
            wait
            CURRENT_JOBS=0
        fi
    done
done

wait

#!/bin/bash

# Define the environment path and requirements file
ENV_PATH="environments/pref_learning"
REQUIREMENTS_FILE="environments/linux_requirements.txt"

# Check if the virtual environment exists, if not, create it
if [ ! -d "$ENV_PATH" ]; then
    echo "Virtual environment not found. Creating it..."
    python3 -m venv "$ENV_PATH"
    source "$ENV_PATH/bin/activate"
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE"
    echo "Environment setup complete."
else
    # Activate the virtual environment if it already exists
    echo "Activating the virtual environment..."
    source "$ENV_PATH/bin/activate"
fi

# Default array of trajectories if none are provided as input
TRAJECTORIES=(1000000 1000000 10000)

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

# Calculate the distribution
length=$((rules + 1))

# Avoid division by zero
if [ "$length" -le 0 ]; then
    echo "Error: The calculated length must be greater than 0."
    exit 1
fi

value=$(bc <<< "scale=10; 1 / $length")

# Create the list of distribution values
distribution=$(printf "%.10g " $(yes "$value" | head -n "$length"))

# Trim trailing whitespace
distribution=$(echo "$distribution" | xargs)

# Fixed parameters
EPOCHS=200
GENERATIONS=100
PARAM_FILE="./best_params.yaml"
MAIN_SCRIPT="main.py"

# Remove any existing zip files for figures and trajectories to avoid conflicts
rm -f figures_*.zip trajectories_*.zip

# Create the zips directory if it doesn't exist
mkdir zips

# Loop over each trajectory value
for TRAJ in "${TRAJECTORIES[@]}"; do
    # Remove the directories to prepare for the next run
    rm -rf figures trajectories

    echo "Running with ${TRAJ} trajectories..."

    # Run the main.py script
    python "$MAIN_SCRIPT" -e "$EPOCHS" -t "$TRAJ" -g "$GENERATIONS" -p "$PARAM_FILE" --rules_distribution "$distribution" --headless

    # Check if the directories exist and zip them
    if [ -d "figures" ]; then
        zip -r "zips/figures_t${TRAJ}_r${rules}.zip" figures
    else
        echo "Warning: figures directory not found for ${TRAJ} trajectories."
    fi

    if [ -d "trajectories" ]; then
        zip -r "zips/trajectories_t${TRAJ}_r${rules}.zip" trajectories
    else
        echo "Warning: trajectories directory not found for ${TRAJ} trajectories."
    fi

    echo "Completed run with ${TRAJ} trajectories."
done

echo "All runs completed."

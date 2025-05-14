#!/bin/bash
#SBATCH --job-name=baseline_r3
#SBATCH --output=baseline_r3_%j.out
#SBATCH --error=baseline_r3_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=24:00:00
#SBATCH --nodes=1                  # Request 3 nodes :contentReference[oaicite:0]{index=0}
#SBATCH --ntasks-per-node=1        # One task on each node :contentReference[oaicite:1]{index=1}
#SBATCH --cpus-per-task=10         # Keep your CPU allocation
#SBATCH --mem=100G                 # Keep your memory allocation
#SBATCH --gres=gpu:1               # One GPU per node :contentReference[oaicite:2]{index=2}
#SBATCH --account=pr_100_tandon_priority

LOG_DIR="/scratch/$USER/ai-car-preference-learning/output"
GPU_LOG_FILE="$LOG_DIR/gpu_used_${SLURM_JOB_ID}.txt"
mkdir -p $LOG_DIR
log_and_email() {
    MESSAGE="$1"
    echo "$MESSAGE" | tee -a "$GPU_LOG_FILE"
    echo -e "Subject:[Slurm Job: $SLURM_JOB_ID] Status Update\n\n$MESSAGE" | sendmail fyy2003@nyu.edu
}
log_and_email "Starting job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"

module purge
module load parallel/20201022

module purge
module load parallel/20201022
cd /scratch/$USER/ai-car-preference-learning
source venv/bin/activate
pip install -r .devcontainer/requirements.txt

if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
if [ ! -f "orientation_data.csv" ]; then python orientation/orientation_data.py; fi
./scripts/run_basic.sh -r 3 -p -h 
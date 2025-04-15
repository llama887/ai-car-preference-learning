#!/bin/bash
#SBATCH --job-name=collect_data_parallel
#SBATCH --output=collect_data_parallel_%j.out
#SBATCH --error=collect_data_parallel_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=12:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --partition=a100_2
#SBATCH --gres=gpu:a100:3
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
cd /scratch/fyy2003/ai-car-preference-learning
source venv/bin/activate
pip install -r .devcontainer/requirements.txt

if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
if [ ! -f "orientation_data.csv" ]; then python orientation/orientation_data.py; fi

CUDA_VISIBLE_DEVICES=0 ./scripts/run_basic.sh -r 3 -p -h &
CUDA_VISIBLE_DEVICES=1 ./scripts/run_basic.sh -r 2 -p -h &
CUDA_VISIBLE_DEVICES=2 ./scripts/run_basic.sh -r 1 -p -h &
wait

python performance_plots.py -c 3


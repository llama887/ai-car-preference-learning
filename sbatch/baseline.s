#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=baseline_%A_%a.out
#SBATCH --error=baseline_%A_%a.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=48:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=1-3
#SBATCH --account=pr_100_tandon_priority

LOG_DIR="/scratch/$USER/ai-car-preference-learning/output"
mkdir -p $LOG_DIR/output/done_flags
GPU_LOG_FILE="$LOG_DIR/gpu_used_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

log_and_email() {
    MESSAGE="$1"
    echo "$MESSAGE" | tee -a "$GPU_LOG_FILE"
    echo -e "Subject:[Slurm Job: $SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID] Status Update\n\n$MESSAGE" | sendmail fyy2003@nyu.edu
}

log_and_email "Starting job array task: $SLURM_ARRAY_TASK_ID"

module purge
module load parallel/20201022
cd /scratch/$USER/ai-car-preference-learning
source venv/bin/activate
pip install -r .devcontainer/requirements.txt

export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

make run_hpc_baseline_task

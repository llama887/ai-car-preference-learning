#!/bin/bash
#SBATCH --job-name=random_init
#SBATCH --array=0-1999
#SBATCH --output=random_init_%A_%a.out
#SBATCH --error=random_init_%A_%a.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=1            
#SBATCH --mem=16G
#SBATCH --account=pr_100_tandon_priority

### -------------------- Logging Setup -------------------- ###
LOG_DIR="/scratch/$USER/ai-car-preference-learning/output"
GPU_LOG_FILE="$LOG_DIR/gpu_used_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
mkdir -p $LOG_DIR
log_and_email() {
    MESSAGE="$1"
    echo "$MESSAGE" | tee -a "$GPU_LOG_FILE"
    echo -e "Subject:[Slurm Job: $SLURM_JOB_ID] Status Update\n\n$MESSAGE" | sendmail fyy2003@nyu.edu
}
log_and_email "Starting array task: $SLURM_ARRAY_TASK_ID of job $SLURM_JOB_ID"

module purge
cd /scratch/$USER/ai-car-preference-learning
source venv/bin/activate

(
  while true; do
    echo "===== CPU usage at $(date) =====" >> /scratch/$USER/ai-car-preference-learning/cpu_usage_${SLURM_ARRAY_TASK_ID}.log
    mpstat -P ALL 1 1 >> /scratch/$USER/ai-car-preference-learning/cpu_usage_${SLURM_ARRAY_TASK_ID}.log
    echo "" >> /scratch/$USER/ai-car-preference-learning/cpu_usage_${SLURM_ARRAY_TASK_ID}.log
    sleep 60
  done
) &

pip install -r /scratch/$USER/ai-car-preference-learning/.devcontainer/requirements.txt

export SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

make collect_data_only_parallel


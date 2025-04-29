#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=48:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G 
#SBATCH --gres=gpu:1
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

make run_baseline
#!/bin/bash
#SBATCH --job-name=collect_pickles
#SBATCH --output=collect_pickles_%j.out
#SBATCH --error=collect_pickles_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=20:00:00 
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=48           
#SBATCH --mem=369G                         
#SBATCH --account=pr_100_tandon_priority

### -------------------- Logging Setup -------------------- ###
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
cd /scratch/fyy2003/ai-car-preference-learning
source venv/bin/activate
make collect_data
make collect_testset
python subsample_state.py
python orientation/orientation_data.py
make run_generate_trueRF_parallel
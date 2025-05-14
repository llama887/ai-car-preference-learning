#!/bin/bash
#SBATCH --job-name=one_rule
#SBATCH --output=one_rule_%j.out
#SBATCH --error=one_rule_%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --time=24:00:00
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1        # One task on each node :contentReference[oaicite:1]{index=1}
#SBATCH --cpus-per-task=10         # Keep your CPU allocation
#SBATCH --mem=128G                 # Keep your memory allocation
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

stdbuf -oL python -u main.py -e 3000 -t 100000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule1 --trajectory trajectories_rule1 -d "1/2" -d "1/2" -i 1 --headless --skip-retrain 2>&1 | tee logs/log_rule1_100000.log &
stdbuf -oL python -u main.py -e 3000 -t 100000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule2 --trajectory trajectories_rule2 -d "1/2" -d "1/2" -i 2 --headless --skip-retrain 2>&1 | tee logs/log_rule2_100000.log &
stdbuf -oL python -u main.py -e 3000 -t 100000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule3 --trajectory trajectories_rule3 -d "1/2" -d "1/2" -i 3 --headless --skip-retrain 2>&1 | tee logs/log_rule3_100000.log &
wait

stdbuf -oL python -u main.py -e 3000 -t 10000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule1 --trajectory trajectories_rule1 -d "1/2" -d "1/2" -i 1 --headless --skip-retrain 2>&1 | tee logs/log_rule1_10000.log &
stdbuf -oL python -u main.py -e 3000 -t 10000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule2 --trajectory trajectories_rule2 -d "1/2" -d "1/2" -i 2 --headless --skip-retrain 2>&1 | tee logs/log_rule2_10000.log &
stdbuf -oL python -u main.py -e 3000 -t 10000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule3 --trajectory trajectories_rule3 -d "1/2" -d "1/2" -i 3 --headless --skip-retrain 2>&1 | tee logs/log_rule3_10000.log &
wait

stdbuf -oL python -u main.py -e 3000 -t 1000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule1 --trajectory trajectories_rule1 -d "1/2" -d "1/2" -i 1 --headless --skip-retrain 2>&1 | tee logs/log_rule1_1000.log &
stdbuf -oL python -u main.py -e 3000 -t 1000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule2 --trajectory trajectories_rule2 -d "1/2" -d "1/2" -i 2 --headless --skip-retrain 2>&1 | tee logs/log_rule2_1000.log &
stdbuf -oL python -u main.py -e 3000 -t 1000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule3 --trajectory trajectories_rule3 -d "1/2" -d "1/2" -i 3 --headless --skip-retrain 2>&1 | tee logs/log_rule3_1000.log &
wait
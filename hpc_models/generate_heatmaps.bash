#!/bin/bash

echo "plotting 1000"
python reward_heatmap_plot.py -m hpc_models/model_1_100000_epochs_1000_pairs_1_rules.pth -r 1 -s 1000 &

python reward_heatmap_plot.py -m hpc_models/model_12_100000_epochs_1000_pairs_2_rules.pth -r 2 -s 1000 &

python reward_heatmap_plot.py -m hpc_models/model_123_100000_epochs_1000_pairs_3_rules.pth -r 3 -s 1000 &

wait

echo "plotting 10000"
python reward_heatmap_plot.py -m hpc_models/model_1_100000_epochs_10000_pairs_1_rules.pth -r 1 -s 10000 &

python reward_heatmap_plot.py -m hpc_models/model_12_100000_epochs_10000_pairs_2_rules.pth -r 2 -s 10000 &

python reward_heatmap_plot.py -m hpc_models/model_123_100000_epochs_10000_pairs_3_rules.pth -r 3 -s 10000 &

wait

echo "plotting 100000"
python reward_heatmap_plot.py -m hpc_models/model_1_100000_epochs_100000_pairs_1_rules.pth -r 1 -s 100000 &

python reward_heatmap_plot.py -m hpc_models/model_12_100000_epochs_100000_pairs_2_rules.pth -r 2 -s 100000 &    

python reward_heatmap_plot.py -m hpc_models/model_123_100000_epochs_100000_pairs_3_rules.pth -r 3 -s 100000 &

Wait

echo "plotting 1000000"
python reward_heatmap_plot.py -m hpc_models/model_1_100000_epochs_1000000_pairs_1_rules.pth -r 1 -s 1000000 &

python reward_heatmap_plot.py -m hpc_models/model_12_100000_epochs_1000000_pairs_2_rules.pth -r 2 -s 1000000 &

python reward_heatmap_plot.py -m hpc_models/model_123_100000_epochs_1000000_pairs_3_rules.pth -r 3 -s 1000000 &

wait




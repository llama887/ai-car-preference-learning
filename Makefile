run_baseline:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	./scripts/run_basic.sh -r 3 -h
	./scripts/run_basic.sh -r 2 -h
	./scripts/run_basic.sh -r 1 -h
	python performance_plots.py -c 3

run_baseline_parallel:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	./scripts/run_basic.sh -r 3 -p -h 
	./scripts/run_basic.sh -r 2 -p -h 
	./scripts/run_basic.sh -r 1 -p -h 
	python performance_plots.py -c 3

run_baseline_parallel_hpc:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	CUDA_VISIBLE_DEVICES=0 ./scripts/run_basic.sh -r 3 -p -h &
	CUDA_VISIBLE_DEVICES=1 ./scripts/run_basic.sh -r 2 -p -h &
	CUDA_VISIBLE_DEVICES=2 ./scripts/run_basic.sh -r 1 -p -h &
	wait
	python performance_plots.py -c 3


run_baseline_with_subsampling:
	mkdir -p logs
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	if [ ! -f "subsampled_gargantuar_1_length.pkl" ]; then python subsample_state.py; fi
	stdbuf -oL python -u main.py -e 10000 -t 1000000 -s 1 -g 1 -p best_params.yaml --headless -c 1 -d "1/2" -d "1/2" -md subsampled_gargantuar_1_length.pkl --heatmap --skip-retrain --trajectory subsampled_trajectories_r1 --figure subsampled_figures_r1 --skip-test-accuracy --model models_subsampled 2>&1 | tee logs/log_1_r_subsample.log
	stdbuf -oL python -u main.py -e 10000 -t 1000000 -s 1 -g 1 -p best_params.yaml --headless -c 2 -d "3/12" -d "3/12" -d "6/12" -md subsampled_gargantuar_1_length.pkl --heatmap --skip-retrain --trajectory subsampled_trajectories_r2 --figure subsampled_figures_r2 --skip-test-accuracy --model models_subsampled 2>&1 | tee logs/log_2_r_subsample.log
	stdbuf -oL python -u main.py -e 10000 -t 1000000 -s 1 -g 1 -p best_params.yaml --headless -c 3 -d "2/12" -d "2/12" -d "2/12" -d "6/12" -md subsampled_gargantuar_1_length.pkl --heatmap --skip-retrain --trajectory subsampled_trajectories_r3 --figure subsampled_figures_r3 --skip-test-accuracy --model models_subsampled 2>&1 | tee logs/log_3_r_subsample.log

run_baseline_and_ensembling:
	# ./scripts/run_basic.sh -r 3
	# ./scripts/run_basic.sh -r 2
	# ./scripts/run_basic.sh -r 1
	./scripts/run_basic.sh -r 3 -e
	./scripts/run_basic.sh -r 2	-e
	./scripts/run_basic.sh -r 1	-e
	python performance_plots.py -c 3 -e

run_baseline_and_ensembling_parallel:
	# ./scripts/run_basic.sh -r 3 -p
	# ./scripts/run_basic.sh -r 2 -p
	# ./scripts/run_basic.sh -r 1 -p
	./scripts/run_basic.sh -r 3 -p -e
	./scripts/run_basic.sh -r 2 -p -e
	./scripts/run_basic.sh -r 1 -p -e
	python performance_plots.py -c 3 -e

run_variable_segments_fixed_pairs:
	./scripts/run_segment_lengths_fixed_pairs.sh -s 6
	python performance_plots.py -c 3 -e

run_variable_segments_fixed_pairs_parallel:
	./scripts/run_segment_lengths_fixed_pairs.sh -s 6 -p
	python performance_plots.py -c 3 -e

run_variable_segment_fixed_duration:
	./scripts/run_segment_lengths_fixed_duration.sh -s 6
	python performance_plots.py -c 3 -e

run_variable_segments_fixed_duration_parallel:
	./scripts/run_segment_lengths_fixed_duration.sh -s 6 -p
	python performance_plots.py -c 3 -e

run_generate_trueRF:
	./scripts/run_trueRF.sh


run_generate_trueRF_parallel:
	./scripts/run_trueRF.sh -p

run_with_partial_rewards:
	./scripts/run_partial_rewards.sh -r 3 -p 3
	python simplex.py

run_with_partial_rewards_iter0:
	python partial_rewards_by_index.py -i 0

run_with_partial_rewards_iter1:
	python partial_rewards_by_index.py -i 1

run_with_partial_rewards_iter2:
	python partial_rewards_by_index.py -i 2

run_with_partial_rewards_iter3:
	python partial_rewards_by_index.py -i 3

run_with_partial_rewards_iter4:
	python partial_rewards_by_index.py -i 4

run_with_partial_rewards_iter5:
	python partial_rewards_by_index.py -i 5

run_with_partial_rewards_iter6:
	python partial_rewards_by_index.py -i 6

run_with_partial_rewards_iter7:
	python partial_rewards_by_index.py -i 7

run_with_partial_rewards_iter8:
	python partial_rewards_by_index.py -i 8

run_with_partial_rewards_iter9:
	python partial_rewards_by_index.py -i 10

run_distribution_by_rules:
# 	python run_distribution_by_rule.py -r 6 -c 1
# 	python run_distribution_by_rule.py -r 6 -c 2
	python run_distribution_by_rule.py -r 6 -c 3
	python distribution_plot_from_data.py

subtract_center_and_split_training_set_into_training_set_and_test_set:
	rm -rf databases/database_gargantuar_testing*
	python database_subtract_center.py
	mv databases/database_gargantuar_1_length databases/database_gargantuar_1_length_poopoo
	mv databases/database_gargantuar_1_length_new databases/database_gargantuar_1_length
	python split_database.py


run_distribution_rule_1:
	python run_distribution_by_rule.py -r 6 -c 1

run_distribution_rule_2:
	python run_distribution_by_rule.py -r 6 -c 2

run_distribution_rule_3_1:
	python run_distribution_by_rule.py -r 6 -c 3 -s 0.1

run_distribution_rule_3_2:
	python run_distribution_by_rule.py -r 6 -c 3 -s 0.2

run_distribution_rule_3_3:
	python run_distribution_by_rule.py -r 6 -c 3 -s 0.3


collect_data:
	rm -rf tmp
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	./scripts/parallel_data_collect.sh -t 20000000 -n 10 
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_1_length.pkl 

collect_testset:
	rm -rf tmp
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	./scripts/parallel_data_collect.sh -t 20000000 -n 10
	python ./combine_gargantuar.py -d tmp -o databases/database_gargantuar_testing_1_length


collect_data_all:
	make collect_data
	make subsample_collect_data_all
	make collect_data_longer_segments

run_hpc_baseline_task:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi

	if [ "$(SLURM_ARRAY_TASK_ID)" = "1" ]; then ./scripts/run_basic.sh -r 1 -p -h; fi
	if [ "$(SLURM_ARRAY_TASK_ID)" = "2" ]; then ./scripts/run_basic.sh -r 2 -p -h; fi
	if [ "$(SLURM_ARRAY_TASK_ID)" = "3" ]; then \
		./scripts/run_basic.sh -r 3 -p -h; \
		python performance_plots.py -c 3; \
	fi

run_one_rule_experiment:
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule1 --trajectory trajectories_rule1 -d "1/2" -d "1/2" -i 1 --headless --skip-retrain 2>&1 | tee logs/log_rule1.log &
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule2 --trajectory trajectories_rule2 -d "1/2" -d "1/2" -i 2 --headless --skip-retrain 2>&1 | tee logs/log_rule2.log &
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule3 --trajectory trajectories_rule3 -d "1/2" -d "1/2" -i 3 --headless --skip-retrain 2>&1 | tee logs/log_rule3.log &

one_rule_sequential:
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule1_1000000 --trajectory trajectories_rule1_1000000 -d "1/2" -d "1/2" -i 1 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule1.log &
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule2_1000000 --trajectory trajectories_rule2_1000000 -d "1/2" -d "1/2" -i 2 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule2.log &
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule3_1000000 --trajectory trajectories_rule3_1000000 -d "1/2" -d "1/2" -i 3 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule3.log &


	stdbuf -oL python -u main.py -e 3000 -t 100000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule1_100000 --trajectory trajectories_rule1_100000 -d "1/2" -d "1/2" -i 1 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule1_100000.log
	stdbuf -oL python -u main.py -e 3000 -t 100000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule2_100000 --trajectory trajectories_rule2_100000 -d "1/2" -d "1/2" -i 2 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule2_100000.log
	stdbuf -oL python -u main.py -e 3000 -t 100000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule3_100000 --trajectory trajectories_rule3_100000 -d "1/2" -d "1/2" -i 3 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule3_100000.log
	

	stdbuf -oL python -u main.py -e 3000 -t 10000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule1_10000 --trajectory trajectories_rule1_10000 -d "1/2" -d "1/2" -i 1 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule1_10000.log
	stdbuf -oL python -u main.py -e 3000 -t 10000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule2_10000 --trajectory trajectories_rule2_10000 -d "1/2" -d "1/2" -i 2 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule2_10000.log
	stdbuf -oL python -u main.py -e 3000 -t 10000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule3_10000 --trajectory trajectories_rule3_10000 -d "1/2" -d "1/2" -i 3 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule3_10000.log
	

	stdbuf -oL python -u main.py -e 3000 -t 1000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule1_1000 --trajectory trajectories_rule1_1000 -d "1/2" -d "1/2" -i 1 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule1_1000.log
	stdbuf -oL python -u main.py -e 3000 -t 1000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule2_1000 --trajectory trajectories_rule2_1000 -d "1/2" -d "1/2" -i 2 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule2_1000.log
	stdbuf -oL python -u main.py -e 3000 -t 1000 -g 200 -p ./best_params.yaml -c 1 --figure figures_rule3_1000 --trajectory trajectories_rule3_1000 -d "1/2" -d "1/2" -i 3 --headless --skip-retrain --model models_one_rule 2>&1 | tee logs/log_rule3_1000.log
	
backup:
	mv zips_baseline_last zips_baseline_delete
	mv zips_ensembling_last zips_ensembling_delete
	mv zips_baseline zips_baseline_last
	mv zips_ensembling zips_ensembling_last
	rm -rf zips_baseline_delete
	rm -rf zips_ensembling_delete

clean:
	rm -rf wandb
	rm -rf __pycache__
	rm -rf figures*
	rm -rf trajectories*
	rm -rf logs
	rm -rf rl_zoo_weights
	rm -rf tmp
	find . -type f -name '*.zip' -delete
	find . -type f -name '*.pth' -delete

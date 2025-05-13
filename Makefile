run_baseline:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	if [ ! -f "orientation_data.csv" ]; then python orientation/orientation_data.py; fi
	./scripts/run_basic.sh -r 3 -h
	./scripts/run_basic.sh -r 2 -h
	./scripts/run_basic.sh -r 1 -h
	python performance_plots.py -c 3

run_baseline_hpc:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi; \
	if [ ! -f "orientation_data.csv" ]; then python orientation/orientation_data.py; fi; \
	srun --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --exclusive ./scripts/run_basic.sh -r 3 -h & \
	srun --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --exclusive ./scripts/run_basic.sh -r 2 -h & \
	srun --nodes=1 --ntasks=1 --cpus-per-task=10 --gres=gpu:1 --exclusive ./scripts/run_basic.sh -r 1 -h & \
	wait; \
	python performance_plots.py -c 3

run_baseline_parallel:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	if [ ! -f "orientation_data.csv" ]; then python orientation/orientation_data.py; fi
	./scripts/run_basic.sh -r 3 -p -h 
	./scripts/run_basic.sh -r 2 -p -h 
	./scripts/run_basic.sh -r 1 -p -h 
	python performance_plots.py -c 3

run_baseline_parallel_hpc:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	if [ ! -f "orientation_data.csv" ]; then python orientation/orientation_data.py; fi
	CUDA_VISIBLE_DEVICES=0 ./scripts/run_basic.sh -r 3 -p -h &
	CUDA_VISIBLE_DEVICES=1 ./scripts/run_basic.sh -r 2 -p -h &
	CUDA_VISIBLE_DEVICES=2 ./scripts/run_basic.sh -r 1 -p -h &
	wait
	python performance_plots.py -c 3



run_baseline_with_subsampling:
	mkdir -p logs
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	if [ ! -f "orientation_data.csv" ]; then python orientation/orientation_data.py; fi
	if [ ! -f "subsampled_gargantuar_1_length.pkl" ]; then python subsample_state.py; fi
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -s 1 -g 1 -p best_params.yaml --headless -c 1 -d "1/2" -d "1/2" -md subsampled_gargantuar_1_length.pkl --heatmap --skip-retrain --trajectory subsampled_trajectories_r1 --figure subsampled_figures_r1 2>&1 | tee logs/log_1_r_subsample.log
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -s 1 -g 1 -p best_params.yaml --headless -c 2 -d "3/12" -d "3/12" -d "6/12" -md subsampled_gargantuar_1_length.pkl --heatmap --skip-retrain --trajectory subsampled_trajectories_r2 --figure subsampled_figures_r2 2>&1 | tee logs/log_2_r_subsample.log
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -s 1 -g 1 -p best_params.yaml --headless -c 3 -d "1/6" -d "1/6" -d "1/6" -d "3/6" -md subsampled_gargantuar_1_length.pkl --heatmap --skip-retrain --trajectory subsampled_trajectories_r3 --figure subsampled_figures_r3 2>&1 | tee logs/log_3_r_subsample.log

run_baseline_with_subsampling_parallel:
	mkdir -p logs
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	if [ ! -f "orientation_data.csv" ]; then python orientation/orientation_data.py; fi
	if [ ! -f "subsampled_gargantuar_1_length.pkl" ]; then python subsample_state.py; fi
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -s 1 -g 1 -p best_params.yaml --headless -c 1 -d "1/2" -d "1/2" -md subsampled_gargantuar_1_length.pkl --heatmap --skip-retrain --trajectory subsampled_trajectories_r1 --figure subsampled_figures_r1 2>&1 | tee logs/log_1_r_subsample.log &
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -s 1 -g 1 -p best_params.yaml --headless -c 2 -d "3/12" -d "3/12" -d "6/12" -md subsampled_gargantuar_1_length.pkl --heatmap --skip-retrain --trajectory subsampled_trajectories_r2 --figure subsampled_figures_r2 2>&1 | tee logs/log_2_r_subsample.log &
	stdbuf -oL python -u main.py -e 3000 -t 1000000 -s 1 -g 1 -p best_params.yaml --headless -c 3 -d "1/6" -d "1/6" -d "1/6" -d "3/6" -md subsampled_gargantuar_1_length.pkl --heatmap --skip-retrain --trajectory subsampled_trajectories_r3 --figure subsampled_figures_r3 2>&1 | tee logs/log_3_r_subsample.log &
	wait

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

run_with_partial_rewards: database_test_2_rules.pkl
	./scripts/run_partial_rewards.sh -r 3 -p 6
	python simplex.py

run_on_subsampled_data:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	./scripts/run_basic.sh -r 3 -h
	./scripts/run_basic.sh -r 2 -h
	./scripts/run_basic.sh -r 1 -h
	python performance_plots.py -c 3


collect_data:
	rm -rf tmp
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	./scripts/parallel_data_collect.sh -t 20000000 -n 20
	python ./combine_gargantuar.py -d tmp -o databases/database_gargantuar_1_length

collect_testset:
	rm -rf tmp
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	./scripts/parallel_data_collect.sh -t 20000000 -n 20
	python ./combine_gargantuar.py -d tmp -o databases/database_gargantuar_testing_1_length


subsample_collect_data_all:
	python subsample_state.py -s 2000000 -r 1
	python subsample_state.py -s 2000000 -r 2
	python subsample_state.py -s 2000000 -r 3

collect_data_longer_segments:
	./scripts/parallel_data_collect.sh -t 20000000 -n 10 -s 2
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_2_length_3_rules_tmp.pkl
	./scripts/parallel_data_collect.sh -t 20000000 -n 10 -s 3
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_3_length_3_rules_tmp.pkl
	./scripts/parallel_data_collect.sh -t 20000000 -n 10 -s 4
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_4_length_3_rules_tmp.pkl
	./scripts/parallel_data_collect.sh -t 20000000 -n 10 -s 5
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_5_length_3_rules_tmp.pkl
	./scripts/parallel_data_collect.sh -t 20000000 -n 10 -s 6
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_6_length_3_rules_tmp.pkl


collect_data_all:
	make collect_data
	make subsample_collect_data_all
	make collect_data_longer_segments

run_hpc_baseline_task:
	if [ ! -f "grid_points.pkl" ]; then python save_gridpoints.py; fi
	if [ ! -f "orientation_data.csv" ]; then python orientation/orientation_data.py; fi

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

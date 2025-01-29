run_baseline:
	./scripts/run_basic.sh -r 3
	./scripts/run_basic.sh -r 2
	./scripts/run_basic.sh -r 1
	python performance_plots.py -c 3


run_baseline_parallel:
	./scripts/run_basic.sh -r 3 -p
	./scripts/run_basic.sh -r 2 -p
	./scripts/run_basic.sh -r 1 -p
	python performance_plots.py -c 3


run_baseline_and_ensembling:
	./scripts/run_basic.sh -r 3
	./scripts/run_basic.sh -r 2
	./scripts/run_basic.sh -r 1
	./scripts/run_basic.sh -r 3 -e
	./scripts/run_basic.sh -r 2	-e
	./scripts/run_basic.sh -r 1	-e
	python performance_plots.py -c 3 -e

run_baseline_and_ensembling_parallel:
	./scripts/run_basic.sh -r 3 -p
	./scripts/run_basic.sh -r 2 -p
	./scripts/run_basic.sh -r 1 -p
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


run_with_subsampling:
	...


run_with_distributions:
	./scripts/run_distribution.sh -r 6


database_test_1_rules.pkl:
	./scripts/parallel_data_collect.sh -t 2000000 -r 1 -n 10 -g
	python ./combine_gargantuar.py -d tmp -o database_test -r 1 -p

database_test_2_rules.pkl:
	./scripts/parallel_data_collect.sh -t 2000000 -r 2 -n 10 -g
	python ./combine_gargantuar.py -d tmp -o database_test -r 2 -p

database_test_3_rules.pkl:
	./scripts/parallel_data_collect.sh -t 2000000 -r 3 -n 10 -g
	python ./combine_gargantuar.py -d tmp -o database_test -r 3 -p

run_with_partial_rewards: database_test_2_rules.pkl
	./scripts/run_partial_rewards.sh -r 3 -p 1
	python simplex.py

collect_data_3_rules:
	./scripts/parallel_data_collect.sh -t 20000000 -r 3 -n 10
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_1_length_3_rules_tmp.pkl

collect_data_2_rules:
	./scripts/parallel_data_collect.sh -t 20000000 -r 2 -n 10
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_1_length_2_rules_tmp.pkl

collect_data_1_rules:
	./scripts/parallel_data_collect.sh -t 20000000 -r 1 -n 10
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_1_length_1_rules_tmp.pkl

clean:
	rm -rf wandb
	rm -rf __pycache__
	rm -rf figures
	rm -rf trajectories
	rm -rf logs
	rm -rf rl_zoo_weights
	rm -rf tmp
	find . -type f -name '*.zip' -delete
	find . -type f -name '*.pth' -delete

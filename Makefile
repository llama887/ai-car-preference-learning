run_baseline:
	./scripts/run_basic.sh -r 3
	./scripts/run_basic.sh -r 2
	./scripts/run_basic.sh -r 1
	./scripts/run_plots.sh -r 3


run_baseline_parallel:
	# ./scripts/run_basic.sh -r 3 -p
	./scripts/run_basic.sh -r 2 -p
	./scripts/run_basic.sh -r 1 -p
	./scripts/run_plots.sh -r 3


run_with_subsampling:
	...

run_with_partial_rewards:
	./scripts/run_partial_rewards.sh -r 5 -p 3
	python simplex.py

run_with_ensemble:
	...

run_pendulum:
	mkdir logs
	python ./replication/train_agent.py
	...

collect_data:
	./scripts/parallel_data_collect.sh -t 10000000 -r 3 -n 10
	python ./combine_gargantuar.py -d tmp -o database_gargantuar_1_length.pkl

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

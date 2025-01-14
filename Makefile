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

clean:
	rm -rf wandb
	rm -rf __pycache__
	rm -rf figures
	rm -rf trajectories
	rm -rf logs
	rm -rf rl_zoo_weights
	find . -type f -name '*.zip' -delete
	find . -type f -name '*.pth' -delete

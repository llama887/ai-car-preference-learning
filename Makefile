run_baseline:
	mkdir logs
	./scripts/run_basic.sh -r 3
	./scripts/run_basic.sh -r 2
	./scripts/run_basic.sh -r 1

run_with_subsampling:
	...

run_with_ensemble:
	...

run_pendulum:
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

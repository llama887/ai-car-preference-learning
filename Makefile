run_baseline:
	make clean
	./scripts/run_basic.sh -r 3
	./scripts/run_basic.sh -r 2
	./scripts/run_basic.sh -r 1

run_with_subsampling:
	make clean
	...

run_with_ensemble:
	make clean
	...

run_pong:
	make clean
	...

clean:
	rm -rf wandb
	rm -rf __pycache__
	rm -rf figures
	rm -rf trajectories
	find . -type f -name '*.zip' -delete
	find . -type f -name '*.pth' -delete


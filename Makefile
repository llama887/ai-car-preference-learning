run_baseline:
	make clean
	./scripts/run_basic.sh -r 1
	./scripts/run_basic.sh -r 2
	./scripts/run_basic.sh -r 3

clean:
	rm -rf wandb
	rm -rf __pycache__
	rm -rf figures
	rm -rf trajectories
	find . -type f -name '*.zip' -delete
	find . -type f -name '*.pth' -delete


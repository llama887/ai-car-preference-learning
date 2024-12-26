run_basic:
	...

clean:
	rm -rf wandb
	rm -rf __pycache__
	rm -rf figures
	rm -rf trajectories
	find . -type f -name '*.zip' -delete
	find . -type f -name '*.pth' -delete


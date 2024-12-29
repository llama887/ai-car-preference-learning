run_baseline:
	make clean
	mkdir logs
	@echo "Running ./scripts/run_basic.sh -r 3" && time ./scripts/run_basic.sh -r 3 2>&1 | tee -a logs/execution_times.log
	@echo "Running ./scripts/run_basic.sh -r 2" && time ./scripts/run_basic.sh -r 2 2>&1 | tee -a logs/execution_times.log
	@echo "Running ./scripts/run_basic.sh -r 1" && time ./scripts/run_basic.sh -r 1 2>&1 | tee -a logs/execution_times.log

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
	rm -rf logs
	find . -type f -name '*.zip' -delete
	find . -type f -name '*.pth' -delete

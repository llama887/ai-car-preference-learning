import argparse
import os
import pickle
import sys

import yaml

import agent
import reward
import reward_heatmap_plot
import rules
from agent import (
    AGENTS_PER_GENERATION,
    generate_database_from_segments,
    load_models,
    run_population,
)
from debug_plots import (
    handle_plotting_rei,
    populate_lists,
)
from reward import (
    train_reward_function,
)
from test_accuracy import(
    test_model,
)

os.environ["WANDB_SILENT"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def start_simulation(
    config_path,
    max_generations,
    number_of_pairs,
    run_type,
    noHead=True,
    use_ensemble=False,
):
    # Set number of trajectories
    agent.number_of_pairs = number_of_pairs

    return (
        run_population(
            config_path=config_path,
            max_generations=max_generations,
            number_of_pairs=number_of_pairs,
            runType=run_type,
            noHead=noHead,
            use_ensemble=use_ensemble,
        ),
        agent.rules_followed,
    )


def parse_to_float(s):
    try:
        return float(s)
    except ValueError:
        try:
            numerator, denominator = map(float, s.split("/"))
            return numerator / denominator
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"Cannot convert '{s}' to float")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )
    parse.add_argument(
        "-e",
        "--epochs",
        type=int,
        nargs=1,
        help="Number of epochs to train the model",
    )
    parse.add_argument(
        "-t",
        "--trajectories",
        type=int,
        nargs=1,
        help="Number of pairs of segments to collect",
    )
    parse.add_argument(
        "-s",
        "--segment",
        type=int,
        help="Length of segments",
    )
    parse.add_argument(
        "-f",
        "--figure",
        type=str,
        help="Figure Folder Name",
    )
    parse.add_argument(
        "--trajectory",
        type=str,
        help="Trajectory Folder Name",
    )
    parse.add_argument(
        "-g",
        "--generations",
        type=int,
        nargs=1,
        help="Number of generations to train the agent",
    )
    parse.add_argument(
        "-p",
        "--parameters",
        type=str,
        help="Directory to hyperparameter yaml file",
    )
    parse.add_argument(
        "--headless", action="store_true", help="Run simulation without GUI"
    )
    parse.add_argument(
        "--ensemble", action="store_true", help="Train an ensemble of 3 predictors"
    )
    parse.add_argument(
        "-r",
        "--reward",
        type=str,
        action="append",
        help="Directory to reward function weights",
    )
    parse.add_argument(
        "-c",
        "--composition",
        type=int,
        help="number of rules",
    )
    parse.add_argument(
        "-d",
        "--distribution",
        type=str,
        action="append",
        help="Distribution of segments collected",
    )
    parse.add_argument(
        "-md",
        "--master-database",
        type=str,
        help="Path to master database",
    )
    parse.add_argument(
        "--heatmap", action="store_true", help="Plot heatmap of collected data"
    )
    parse.add_argument("--skip-plots", action="store_true", help="Skip debug plots")

    args = parse.parse_args()

    if args.master_database:
        agent.master_database = args.master_database

    if (
        (args.trajectories is not None and args.trajectories[0] < 0)
        or (args.generations is not None and args.generations[0] < 0)
        or (args.epochs is not None and args.epochs[0] < 0)
    ):
        print("Invalid input. All arguments must be positive integers.")
        sys.exit(1)

    # Display simulator or not
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if args.trajectory:
        agent.trajectories_path = args.trajectory + "/"
        os.makedirs(agent.trajectories_path, exist_ok=True)
    trajectory_path = agent.trajectories_path

    # Set segment length
    if args.segment and args.segment < 1:
        raise Exception("Can not have segments with length < 1")
    agent.train_trajectory_length = args.segment if args.segment else 1

    # Set distribution of segment rule satisfaction
    if args.composition and args.trajectories:
        rules.NUMBER_OF_RULES = args.composition
        num_pairs = 10 * args.trajectories[0] if args.ensemble else args.trajectories[0]
        database_path = f"{agent.trajectories_path}database_{num_pairs}_pairs_{args.composition}_rules_{agent.train_trajectory_length}_length.pkl"
    else:
        print("Missing either -c flag or -t flag")

    if args.distribution:
        if args.master_database and "subsampled" in args.master_database:
            with open(args.master_database, "rb") as file:
                data = pickle.load(file)
                total_segments = 0
                for segment_per_rule in data:
                    total_segments += len(segment_per_rule)

                rules.SEGMENT_DISTRIBUTION_BY_RULES = [
                    len(s) / total_segments for s in data
                ]
        else:
            try:
                rules.SEGMENT_DISTRIBUTION_BY_RULES = [
                    parse_to_float(d) for d in args.distribution
                ]
            except Exception:
                print(
                    "Distribution input too advanced for Alex and Franklin's caveman parser. (or maybe you input something weird sry)"
                )
                sys.exit()
            sum_dist = sum(rules.SEGMENT_DISTRIBUTION_BY_RULES)
            rules.SEGMENT_DISTRIBUTION_BY_RULES = [
                d / sum_dist for d in rules.SEGMENT_DISTRIBUTION_BY_RULES
            ]
            assert (
                len(rules.SEGMENT_DISTRIBUTION_BY_RULES) == rules.NUMBER_OF_RULES + 1
            ), (
                f"SEGMENT_DISTRIBUTION_BY_RULES: {rules.SEGMENT_DISTRIBUTION_BY_RULES} does not have one more than the length specified in NUMBER_OF_RULES: {rules.NUMBER_OF_RULES}"
            )
            assert sum(rules.SEGMENT_DISTRIBUTION_BY_RULES) == 1, (
                f"SEGMENT_DISTRIBUTION_BY_RULES: {rules.SEGMENT_DISTRIBUTION_BY_RULES} does not sum to 1 (even after scaling)"
            )

    if args.segment and args.segment < 1:
        raise Exception("Can not have segments with length < 1")
    agent.train_trajectory_length = args.segment if args.segment else 1

    model_weights = ""
    if args.reward is None:
        if args.master_database and "subsampled" in args.master_database:
            with open(args.master_database, "rb") as file:
                data = pickle.load(file)
                generate_database_from_segments(
                    data[: rules.NUMBER_OF_RULES + 1], args.trajectories[0]
                )
        else:
            # start the simulation in data collecting mode
            num_traj, collecting_rules_followed = start_simulation(
                "./config/data_collection_config.txt",
                args.trajectories[0],
                args.trajectories[0],
                "collect",
                args.headless,
                args.ensemble,
            )

        print("Starting training on trajectories...")
        print(
            f"train_reward_function({database_path}, {args.epochs[0]}, {args.parameters}, {args.ensemble}, {args.figure})"
        )
        train_reward_function(
            trajectories_file_path=database_path,
            epochs=args.epochs[0],
            parameters_path=args.parameters,
            use_ensemble=args.ensemble,
            figure_folder_name=args.figure,
            return_stat=None,
        )

        print("Finished training model...")

        if not args.parameters:
            sys.exit()
        # run the simulation with the trained reward function
        if args.ensemble:
            model_weights = ["QUICK", reward.ensemble_path]
        else:
            model_weights = [(reward.models_path + f"model_{args.epochs[0]}_epochs_{args.trajectories[0]}_pairs_{rules.NUMBER_OF_RULES}_rules.pth")]
    else:
        model_weights = args.reward

    # run the simulation with the true reward function (if trajectories do not exist yet)
    if os.path.exists(
        f"trueRF_trajectories/trueRF_{args.generations * AGENTS_PER_GENERATION}_trajectories_{rules.NUMBER_OF_RULES}_rules.pkl"
    ):
        truePairs = args.generations * AGENTS_PER_GENERATION
    else:
        print("Simulating on true reward function...")
        truePairs, true_rules_followed = start_simulation(
            "./config/agent_config.txt",
            args.generations[0],
            0,
            "trueRF",
            args.headless,
            args.ensemble,
        )

    # run simulation on trained reward function
    with open(
        args.parameters if args.parameters is not None else "best_params.yaml", "r"
    ) as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]
        batch_size = data["batch_size"]

    print("Simulating on trained reward function...")
    load_models(model_weights, hidden_size)

    trainedPairs, trained_rules_followed = start_simulation(
        "./config/agent_config.txt",
        args.generations[0],
        0,
        "trainedRF",
        args.headless,
        args.ensemble,
    )
    if not args.skip_plots:
        true_database = f"trueRF_trajectories/trueRF_{truePairs}_trajectories_{rules.NUMBER_OF_RULES}_rules.pkl"
        trained_database = (
            trajectory_path
            + f"trainedRF_{trainedPairs}_trajectories_{rules.NUMBER_OF_RULES}_rules.pkl"
        )

        model_info = {
            "net": agent.reward_network,
            "ensemble": agent.ensemble,
            "hidden-size": hidden_size,
            "epochs": -1 if args.epochs is None else args.epochs[0],
            "pairs-learned": -1 if args.trajectories is None else args.trajectories[0],
            "agents-per-generation": 20,
        }

        (
            true_agent_satisfaction_segments,
            true_agent_rewards,
            trained_agent_satisfaction_segments,
            trained_agent_rewards,
            trained_segment_rules_satisifed,
            trained_segment_rewards,
            trained_segment_distances,
            training_segment_rules_satisfied,
            training_segment_rewards,
            training_segment_distances,
        ) = populate_lists(
            true_database,
            trained_database,
            database_path,
            model_info,
        )

        print("PLOTTING...")
        handle_plotting_rei(
            model_info,
            true_agent_satisfaction_segments,
            true_agent_rewards,
            trained_agent_satisfaction_segments,
            trained_agent_rewards,
            trained_segment_rules_satisifed,
            trained_segment_rewards,
            trained_segment_distances,
            training_segment_rules_satisfied,
            training_segment_rewards,
            training_segment_distances,
        )
    else:
        print("Plotting skipped.")

    if args.heatmap:
        reward_heatmap_plot.plot_reward_heatmap(
            samples=reward_heatmap_plot.get_samples(
                args.parameters if args.parameters is not None else "best_params.yaml",
                "grid_points.pkl",
            ),
            number_of_rules=args.composition,
            reward_model=agent.reward_network,
            figure_path=reward.figure_path,
        )

    
    output_file = f"{trajectory_path}/test_accuracy.pkl"
    test_file = f"database_test_{rules.NUMBER_OF_RULES}_rules.pkl"
    test_acc, adjusted_test_acc = test_model(model_weights, test_file, hidden_size, batch_size)
    with open(output_file, "wb") as f:
        pickle.dump((test_acc, adjusted_test_acc), f)
    


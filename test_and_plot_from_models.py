import argparse
import glob
import os
import pickle
import re
import yaml
import uuid

import matplotlib
import matplotlib.pyplot as plt

from prettytable import PrettyTable
from test_accuracy import test_model_light

matplotlib.use("Agg")
import numpy as np
import scipy.stats as stats

from agent import AGENTS_PER_GENERATION, load_models
import rules

TEST_ACC_DIR = "test_accs/"
os.makedirs(TEST_ACC_DIR, exist_ok=True)

BASELINE_DIR = "models/"
DISTRIBUTION_DIR = "models_dist_exp/"

T_VALUE_95 = stats.t.ppf((1 + 0.95) / 2, df=19)

yticks = [0.9999, 0.999, 0.99, 0.9, 0.5]
logged_yticks = [-np.log10(-np.log10(y)) if y != 1 else 1000 for y in yticks]


def use_tex():
    plt.rcParams.update({
        "text.usetex": True,
        'font.family': 'serif',
        'font.serif': 'Times',
        'pgf.preamble': r'\usepackage{amsmath,times}',
        'pgf.texsystem': 'pdflatex',
        'backend': 'pgf',
        'figure.figsize': (4.5, 4.5)
    })

def get_distribution_models():
    models_by_satis = {}
    model_pattern = re.compile(r'model_([\d\.]+)_')

    model_files = glob.glob(os.path.join(DISTRIBUTION_DIR, "**", "*.pth"), recursive=True)

    for model_path in model_files:
        match = model_pattern.search(os.path.basename(model_path))
        if match:
            satis = float(match.group(1))
            models_by_satis[satis] = model_path

    return models_by_satis

def get_organized_models():
    """
    Scans the BASELINE_DIR for model files and organizes them by rules and pairs.
    Returns a nested dictionary where the first key is the number of rules,
    the second key is the number of pairs, and the value is the model file path.
    """

    models_by_rules_and_pairs = {}
    model_pattern = re.compile(r'model_.*?_(\d+)_pairs_(\d+)_rules\.pth')
    
    model_files = glob.glob(os.path.join(BASELINE_DIR, "**", "*.pth"), recursive=True)
    
    for model_path in model_files:
        match = model_pattern.search(os.path.basename(model_path))
        if match:
            pairs = int(match.group(1))
            num_rules = int(match.group(2))
            
            if num_rules not in models_by_rules_and_pairs:
                models_by_rules_and_pairs[num_rules] = {}
            
            models_by_rules_and_pairs[num_rules][pairs] = model_path
    
    return models_by_rules_and_pairs

def plot_baseline():
    id = uuid.uuid4()
    models_by_rules_and_pairs = get_organized_models()
    
    print(f"Found models for {len(models_by_rules_and_pairs)} different rule counts")

    with open("best_params.yaml", "r") as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]
        batch_size = data["batch_size"]

    adjusted_test_accs = {}
    for num_rules, pairs_dict in sorted(models_by_rules_and_pairs.items()):
        rules.NUMBER_OF_RULES = num_rules
        rules.RULES_INCLUDED = [i for i in range(1, rules.NUMBER_OF_RULES + 1)]
        if num_rules not in adjusted_test_accs:
            adjusted_test_accs[num_rules] = []

        for pairs, model_path in sorted(pairs_dict.items()):
            print(f"Testing model with {num_rules} rules and {pairs} pairs: {model_path}")        
            test_acc, adjusted_test_acc = test_model_light(
                [model_path], hidden_size, batch_size, violin_name=f"test_baseline_{num_rules}_rules_{pairs}_pairs"
            )

            print(f"Rules: {num_rules}, Pairs: {pairs}, Test Acc: {test_acc:.4f}, Adjusted Test Acc: {adjusted_test_acc:.4f}")
            adjusted_test_accs[num_rules].append((pairs, adjusted_test_acc))
    
    plt.figure(figsize=(10, 6))
    for num_rules, accs in adjusted_test_accs.items():
        pairs, values = zip(*sorted(accs))
        plt.plot(pairs, values, marker='o', label=f"{num_rules} Rules")
    
    plt.xlabel("Number of Pairs")
    plt.xscale('log')
    plt.ylabel("Adjusted Test Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"baseline_test_acc_{id}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for num_rules, accs in adjusted_test_accs.items():
        pairs, values = zip(*sorted(accs))
        logged_accs = [-np.log10(-np.log10(acc)) for acc in values]
        plt.plot(pairs, logged_accs, marker='o', label=f"{num_rules} Rules")

    plt.xlabel("Number of Pairs")
    plt.xscale('log')
    plt.ylabel("Adjusted Test Accuracy")
    plt.yticks(logged_yticks, yticks)
    plt.legend()
    plt.grid()
    plt.savefig(f"baseline_test_acc_logged_{id}.png", dpi=300)
    plt.close()

    with open(f"{TEST_ACC_DIR}baseline_test_acc_{id}.pkl", "wb") as f:
        pickle.dump(adjusted_test_accs, f)

def plot_distribution():
    id = uuid.uuid4()
    models_by_satis = get_distribution_models()

    rules.NUMBER_OF_RULES = 3
    rules.RULES_INCLUDED = [i for i in range(1, rules.NUMBER_OF_RULES + 1)]
    
    print(f"Found {len(models_by_satis)} models for different satisfaction levels")
    
    adjusted_test_accs = []
    with open("best_params.yaml", "r") as file:
        data = yaml.safe_load(file)
        hidden_size = data["hidden_size"]
        batch_size = data["batch_size"]

    for satis, model_path in sorted(models_by_satis.items()):

        print(f"Testing model with satisfaction {satis}: {model_path}")
        test_acc, adjusted_test_acc = test_model_light(
            [model_path], hidden_size, batch_size
        )

        print(f"Satisfaction: {satis}, Test Acc: {test_acc:.4f}, Adjusted Test Acc: {adjusted_test_acc:.4f}")
        adjusted_test_accs.append((satis, adjusted_test_acc))
    
    adjusted_test_accs.sort(key=lambda x: x[0])
    satis_levels, accs = zip(*adjusted_test_accs)

    plt.figure(figsize=(10, 6))
    plt.plot(satis_levels, accs, marker='o')
    plt.xlabel("% of Segments Satisfying All Rules")
    plt.ylabel("Adjusted Test Accuracy")
    plt.grid()
    plt.savefig(f"distribution_test_acc_{id}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    logged_accs = [-np.log10(-np.log10(acc)) for acc in accs]
    plt.plot(satis_levels, logged_accs, marker='o')

    plt.xlabel("% of Segments Satisfying All Rules")
    plt.ylabel("Adjusted Test Accuracy")
    plt.yticks(logged_yticks, yticks)
    plt.legend()
    plt.grid()
    plt.savefig(f"distribution_test_acc_logged_{id}.png", dpi=300)
    plt.close()

    with open(f"{TEST_ACC_DIR}distribution_test_acc_{id}.pkl", "wb") as f:
        pickle.dump(adjusted_test_accs, f)

def replot_baseline(baseline_test_acc_file="baseline_test_acc.pkl"):
    id = baseline_test_acc_file.split("_")[-1].split(".")[0]
    with open(baseline_test_acc_file, "rb") as f:
        adjusted_test_accs = pickle.load(f)

    plt.figure(figsize=(10, 6))
    for num_rules, accs in adjusted_test_accs.items():
        pairs, values = zip(*sorted(accs))
        plt.plot(pairs, values, marker='o', label=f"{num_rules} Rules")

    plt.xlabel("Number of Pairs")
    plt.xscale('log')
    plt.ylabel("Adjusted Test Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(f"baseline_test_acc_{id}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for num_rules, accs in adjusted_test_accs.items():
        pairs, values = zip(*sorted(accs))
        logged_accs = [-np.log10(-np.log10(acc)) for acc in values]
        plt.plot(pairs, logged_accs, marker='o', label=f"{num_rules} Rules")

    plt.xlabel("Number of Pairs")
    plt.xscale("log")
    plt.ylabel("Adjusted Test Accuracy")
    plt.yticks(logged_yticks, yticks)
    plt.legend()
    plt.grid()
    plt.savefig(f"baseline_test_acc_logged_{id}.png", dpi=300)
    plt.close()

def replot_distribution(distribution_test_acc_file="test_accs/distribution_test_acc.pkl"):
    id = distribution_test_acc_file.split("_")[-1].split(".")[0]
    with open(distribution_test_acc_file, "rb") as f:
        adjusted_test_accs = pickle.load(f)

    satis_levels, accs = zip(*adjusted_test_accs)
    table = PrettyTable()
    table.field_names = ["Satisfaction %", "Mean Accuracy"]
    for satis, acc in adjusted_test_accs:
        table.add_row([satis, f"{acc:.4f}"])
    print(table)

    plt.figure(figsize=(10, 6))
    plt.plot(satis_levels, accs, marker='o')
    plt.xlabel("% of Segments Satisfying All Rules")
    plt.ylabel("Adjusted Test Accuracy")
    plt.grid()
    plt.savefig(f"distribution_test_acc_{id}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    logged_accs = [-np.log10(-np.log10(acc)) for acc in accs]
    plt.plot(satis_levels, logged_accs, marker='o')

    plt.xlabel("% of Segments Satisfying All Rules")
    plt.ylabel("Adjusted Test Accuracy")
    plt.yticks(logged_yticks, yticks)
    plt.legend()
    plt.grid()
    plt.savefig(f"distribution_test_acc_logged_{id}.png", dpi=300)
    plt.close()

def plot_baseline_distribution():
    all_adjusted_test_accs = []
    for file in os.listdir(TEST_ACC_DIR):
        if file.startswith("baseline_test_acc") and file.endswith(".pkl"):
            with open(os.path.join(TEST_ACC_DIR, file), "rb") as f:
                adjusted_test_accs = pickle.load(f)
                all_adjusted_test_accs.append(adjusted_test_accs)

    # Plot the distribution of adjusted test accuracies
    plt.figure(figsize=(10, 6))
    aggregate_points = {}
    for adjusted_test_accs in all_adjusted_test_accs:
        for num_rules, accs in adjusted_test_accs.items():
            if num_rules not in aggregate_points:
                aggregate_points[num_rules] = {}
            for pairs in accs:
                num_pairs, acc_value = pairs
                if num_pairs not in aggregate_points[num_rules]:
                    aggregate_points[num_rules][num_pairs] = []
                aggregate_points[num_rules][num_pairs].append(acc_value)
                   
    table = PrettyTable()
    table.field_names = ["Number of Rules", "Pairs", "Mean Accuracy", "Error (95% CI)"]

    # Calculate mean and standard error for each group
    for num_rules in aggregate_points:
        pairs_list = []
        means = []
        errors = []
        
        for num_pairs, values in sorted(aggregate_points[num_rules].items()):
            if len(values) > 0:
                pairs_list.append(num_pairs)
                mean = np.mean(values)
                error = np.std(values) / np.sqrt(len(values)) * T_VALUE_95
                means.append(mean)
                # Calculate standard error (for confidence intervals)
                errors.append(error)
                table.add_row([num_rules, num_pairs, f"{mean:.4f}", f"{error:.4f}"])
        
        # Plot with error bars
        plt.errorbar(pairs_list, means, yerr=errors, marker='o', 
                     linestyle='-', capsize=5, capthick=1, ecolor='black', label=f"{num_rules} Rules")
    
    print(table)
    table.sortby = "Pairs"
    print(table)

    plt.xlabel("Number of Pairs")
    plt.xscale('log')
    plt.ylabel("Adjusted Test Accuracy")
    plt.title("Baseline Performance with 95% Confidence Intervals")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"baseline_distribution_mean.png", dpi=300)
    plt.close()
    
    # Also plot with logged y-axis for better visualization
    plt.figure(figsize=(10, 6))
    for num_rules in aggregate_points:
        pairs_list = []
        means = []
        errors = []
        
        for num_pairs, values in sorted(aggregate_points[num_rules].items()):
            if len(values) > 0:
                pairs_list.append(num_pairs)
                # Transform to logged scale
                means.append(np.mean(values))
                errors.append(np.std(values) / np.sqrt(len(values)) * T_VALUE_95)
        
        logged_means = [-np.log10(-np.log10(m)) for m in means]
        logged_errors = [[logged_means[i] - (-np.log10(-np.log10(means[i] - e))) for i, e in enumerate(errors)], [-np.log10(-np.log10(min(0.999999999, means[i] + e))) - logged_means[i] for i, e in enumerate(errors)]]
        plt.errorbar(pairs_list, logged_means, yerr=logged_errors, marker='o', 
                     linestyle='-', capsize=5, capthick=1, ecolor='black', label=f"{num_rules} Rules")

    plt.xlabel("Number of Pairs")
    plt.xscale('log')
    plt.ylabel("Adjusted Test Accuracy (Logged Scale)")
    plt.yticks(logged_yticks, yticks)
    plt.title("Baseline Performance Logged")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"baseline_distribution_mean_logged.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and plot models.")
    parser.add_argument("-b", "--baseline", type=str, help="Path to the baseline directory.")
    parser.add_argument("-rb", "--replot-baseline", action="store_true", help="Replot baseline from existing files")
    parser.add_argument("-d", "--distribution", type=str, help="Path to the distribution directory.")
    parser.add_argument("-rd", "--replot-distribution", action="store_true", help="Replot distribution from existing files")

    args = parser.parse_args()
    if args.baseline:
        BASELINE_DIR = args.baseline
        plot_baseline()
    if args.distribution:
        DISTRIBUTION_DIR = args.distribution
        plot_distribution()
    if args.replot_baseline:
        replot_baseline()
    if args.replot_distribution:
        replot_distribution()

    plot_baseline_distribution()
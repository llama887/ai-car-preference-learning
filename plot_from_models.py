import argparse
import glob
import os
import pickle
import re
import yaml

import matplotlib
import matplotlib.pyplot as plt

from test_accuracy import test_model_light

matplotlib.use("Agg")
import numpy as np
import scipy.stats as stats

from agent import AGENTS_PER_GENERATION, load_models
import 

BASELINE_DIR = "models/"

T_VALUE_95 = stats.t.ppf((1 + 0.95) / 2, df=19)

yticks = [1, 0.999, 0.99, 0.9, 0.8, 0.6]
logged_yticks = [-np.log10(-np.log10(y)) if y != 1 else 1 for y in yticks]


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
            rules = int(match.group(2))
            
            if rules not in models_by_rules_and_pairs:
                models_by_rules_and_pairs[rules] = {}
            
            models_by_rules_and_pairs[rules][pairs] = model_path
    
    return models_by_rules_and_pairs

def plot_baseline():
    models_by_rules_and_pairs = get_organized_models()
    
    print(f"Found models for {len(models_by_rules_and_pairs)} different rule counts")
    
    adjusted_test_accs = {}
    for rules, pairs_dict in sorted(models_by_rules_and_pairs.items()):
        
        if rules not in adjusted_test_accs:
            adjusted_test_accs[rules] = []
        for pairs, model_path in sorted(pairs_dict.items()):
            with open("best_params.yaml", "r") as file:
                data = yaml.safe_load(file)
                hidden_size = data["hidden_size"]
                batch_size = data["batch_size"]

            print(f"Testing model with {rules} rules and {pairs} pairs: {model_path}")        
            test_acc, adjusted_test_acc = test_model_light(
                [model_path], hidden_size, batch_size
            )

            print(f"Rules: {rules}, Pairs: {pairs}, Test Acc: {test_acc:.4f}, Adjusted Test Acc: {adjusted_test_acc:.4f}")
            adjusted_test_accs[rules].append((pairs, adjusted_test_acc))
    
    plt.figure(figsize=(10, 6))
    for rules, accs in adjusted_test_accs.items():
        pairs, values = zip(*sorted(accs))
        plt.plot(pairs, values, marker='o', label=f"{rules} Rules")
    
    plt.xlabel("Number of Pairs")
    plt.ylabel("Adjusted Test Accuracy")
    plt.title("Baseline Model Performance by Number of Rules and Pairs")
    plt.legend()
    plt.grid()
    plt.savefig("baseline_test_acc.png")
    plt.close()

    with open("baseline_test_acc.pkl", "wb") as f:
        pickle.dump(adjusted_test_accs, f)


if __name__ == "__main__":
    plot_baseline()
    
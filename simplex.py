import plotly.figure_factory as ff
import numpy as np
import glob
import pickle
import os

if __name__ == "__main__":
    distributions = [[] for i in range(3)]
    val_accs = []
    adjusted_val_accs = []

    directories = glob.glob("trajectories_partial_*")
    print(directories)
    for directory in sorted(directories):
        if os.path.isdir(directory): 
            pickle_path = os.path.join(directory, "result.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    result = pickle.load(f)
                    a, b, c, accs = result
                    val_acc = accs["final_validation_acc"]
                    adjusted_val_acc = accs["final_adjusted_validation_acc"]
                    print(f"{'0 rule:':<12}{a:<15} | {'1 rule:':<12}{b:<15} | {'2 rule:':<12}{c:<15} | {'Val acc:':<12}{val_acc:<15.10f} | {'Adjusted Val acc:':<12}{adjusted_val_acc:<15.10f}")
                    distributions[0].append(a)
                    distributions[1].append(b)
                    distributions[2].append(c)
                    val_accs.append(val_acc)
                    adjusted_val_accs.append(adjusted_val_acc)

    fig = ff.create_ternary_contour(np.array(distributions), np.array(val_accs),
                                pole_labels=['0 Rule', '1 Rule', '2 Rule'],
                                interp_mode='cartesian',
                                showscale=True,
                                title=dict(
                                  text='Final Validation for Various Database Distributions'
                                ))
    fig.write_image("simplex.png")

    fig = ff.create_ternary_contour(np.array(distributions), np.array(adjusted_val_accs),
                                pole_labels=['0 Rule', '1 Rule', '2 Rule'],
                                interp_mode='cartesian',
                                showscale=True,
                                title=dict(
                                  text='Final Validation for Various Database Distributions'
                                ))
    fig.write_image("simplex_adjusted.png")
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
import numpy as np
import glob
import pickle
import os

if __name__ == "__main__":
    distributions = [[] for i in range(3)]
    adjusted_test_accs = []

    directories = glob.glob("trajectories_partial/*")
    print(directories)
    for directory in sorted(directories):
        if os.path.isdir(directory): 
            pickle_path = os.path.join(directory, "result.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    result = pickle.load(f)
                    a, b, c, accs = result
                    test_acc = accs["final_test_acc"]
                    adjusted_test_acc = accs["final_adjusted_test_acc"]

                    print(f"{'0 rule:':<12}{a:<15} | {'1 rule:':<12}{b:<15} | {'2 rule:':<12}{c:<15} | {'Test acc:':<12}{test_acc:<15.10f} | {'Adjusted Test acc:':<12}{adjusted_test_acc:<15.10f}")
                    distributions[0].append(a)
                    distributions[1].append(b)
                    distributions[2].append(c)
                    adjusted_test_accs.append(adjusted_test_acc)

    os.makedirs("simplex/", exist_ok=True)
    fig = ff.create_ternary_contour(np.array(distributions), np.array(adjusted_test_accs),
                                pole_labels=['0 Rule', '1 Rule', '2 Rule'],
                                interp_mode='cartesian',
                                showscale=True,
                                )
    fig.write_image("simplex/simplex_test_1.png")

    # fig = ff.create_ternary_contour(np.array(distributions), np.array(adjusted_val_accs),
    #                             pole_labels=['0 Rule', '1 Rule', '2 Rule'],
    #                             interp_mode='cartesian',
    #                             showscale=True,
    #                             )
    # fig.write_image("simplex_validation_1.png")

    df = pd.DataFrame({'0 Rule': distributions[0], '1 Rule': distributions[1], '2 Rule': distributions[2], "test_acc": adjusted_test_accs})
    fig = px.scatter_ternary(df, a="0 Rule", b="1 Rule", c="2 Rule", color="test_acc", color_continuous_scale="Rainbow")
    fig.write_image("simplex/simplex_test_2.png")

    # fig = px.scatter_ternary(df, a="zero_rule", b="one_rule", c="two_rule", color="val_acc", color_continuous_scale="Rainbow")
    # fig.write_image("simplex_validation_2.png")
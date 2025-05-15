import os
import pickle


from run_distribution import plot_data

figure_folder = "distribution_experiment/"
data_points = {}
for i in range(1, 4):
    with open(
            figure_folder + f"output_{i}.pkl",
            "rb",
        ) as f:
            data_points[i] = pickle.load(f)

plot_data(figure_folder, data_points)
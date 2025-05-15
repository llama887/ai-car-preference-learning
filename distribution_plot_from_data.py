import os
import pickle


from run_distribution import plot_data

figure_folder = "distribution_experiment/"
with open(
        figure_folder + "output.pkl",
        "rb",
    ) as f:
        data_points = pickle.load(f)

plot_data(figure_folder, data_points)
import pickle

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


with open("subsampled_trajectories_r1/subsample_database_1000896_pairs_1_rules_1_length.pkl", "rb") as f:
    data = pickle.load(f)

xy_counts = {}

for tuple in data:
    s1, s2, _, _, _ = tuple
    x1, y1 = int(s1[0].position[0]) // 50, int(s1[0].position[1]) // 50
    x2, y2 = int(s1[1].position[0]) // 50, int(s1[1].position[1]) // 50
    xy_counts[(x1, y1)] = 1 + xy_counts.get((x1, y1), 0)
    xy_counts[(x2, y2)] = 1 + xy_counts.get((x2, y2), 0)

# Extract unique x, y pairs and their counts
unique_xy_pairs = list(xy_counts.keys())
counts = list(xy_counts.values())

# Extract separate lists of x, y coordinates for plotting
x_unique, y_unique = zip(*unique_xy_pairs)

# Create a scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_unique, y_unique, c=counts, cmap="viridis", s=100)

# Add a color bar to indicate the number of occurrences
colorbar = plt.colorbar(scatter)
colorbar.set_label("Number of Occurrences")

# Add labels and title
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.title("Scatter Plot with Point Color Indicating Number of Occurrences")
plt.savefig("xy_distribution.png", dpi=600)
plt.close()

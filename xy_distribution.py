import pickle

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")




def plot_xy_from_trajectory_data_pkl(data, filename="xy_distribution.png"):
    """
    Extracts x, y coordinates from the data and plots them as a scatter plot.
    The color of each point indicates the number of occurrences of that (x, y) pair.
    """
    xy_counts = {}

    BIN_SIZE = 25  # Size of each bin in pixels
    for tuple in data:
        s1, s2, _, _, _ = tuple
        x1, y1 = int(s1[0].position[0]) // BIN_SIZE, int(s1[0].position[1]) // BIN_SIZE
        # x2, y2 = int(s1[1].position[0]) // BIN_SIZE, int(s1[1].position[1]) // BIN_SIZE
        xy_counts[(x1, y1)] = 1 + xy_counts.get((x1, y1), 0)
        # xy_counts[(x2, y2)] = 1 + xy_counts.get((x2, y2), 0)

    plot_xy_counts(xy_counts, filename)

def plot_xy_from_segments(segments, filename="xy_distribution.png"):
    """
    Extracts x, y coordinates from the segments and plots them as a scatter plot.
    The color of each point indicates the number of occurrences of that (x, y) pair.
    """
    xy_counts = {}

    BIN_SIZE = 25  # Size of each bin in pixels
    for segment in segments:
        x1, y1 = int(segment[0].position[0]) // BIN_SIZE, int(segment[0].position[1]) // BIN_SIZE
        # x2, y2 = int(segment[1].position[0]) // BIN_SIZE, int(segment[1].position[1]) // BIN_SIZE
        xy_counts[(x1, y1)] = 1 + xy_counts.get((x1, y1), 0)
        # xy_counts[(x2, y2)] = 1 + xy_counts.get((x2, y2), 0)

    plot_xy_counts(xy_counts, filename)

def plot_xy_counts(xy_counts, filename="xy_distribution.png"):
    """
    Plots the x, y coordinates and their counts as a scatter plot.
    """
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
    plt.savefig(filename, dpi=600)
    plt.close()

import copy
import orientation.get_orientation

if __name__ == "__main__":
    with open("subsampled_trajectories_r1"
    "/subsample_database_1000800_pairs_1_rules_1_length.pkl", "rb") as f:
        data = pickle.load(f)
    # plot_xy_from_trajectory_data_pkl(data, "xy_distribution.png")

    segments = []
    for trajectory_pair in data:
        # --- make every cell an independent copy -------------------------------
        first_element_first_pair   = copy.deepcopy(trajectory_pair[0][0])
        first_element_second_pair  = copy.deepcopy(trajectory_pair[0][1])
        second_element_first_pair  = copy.deepcopy(trajectory_pair[1][0])
        second_element_second_pair = copy.deepcopy(trajectory_pair[1][1])

        # --- subtract the circle centre once for each distinct pair -----------
        centre_x, centre_y = orientation.get_orientation.CIRCLE_CENTER
        for state_action_pair in (
            first_element_first_pair,
            first_element_second_pair,
            second_element_first_pair,
            second_element_second_pair,
        ):
            state_action_pair.position[0] -= centre_x
            state_action_pair.position[1] -= centre_y

        # --- rebuild the two‑by‑two structure ---------------------------------
        trajectory_pair_copy = (
            (first_element_first_pair,  first_element_second_pair),
            (second_element_first_pair, second_element_second_pair),
        )

        
        if trajectory_pair_copy[0][0].position[0] // 25 < -30 or trajectory_pair_copy[0][0].position[1] // 25 < -30:
            new_tp = copy.deepcopy(trajectory_pair)
            print("Initial Position:", trajectory_pair[0][0].position, "\n", trajectory_pair_copy[0][0].position[0] // 25, "\n", trajectory_pair_copy[0][0].position[1] // 25)
            print("Post Centering Positions:", trajectory_pair_copy[0][0].position)
            print("CIRCLE_CENTER:", orientation.get_orientation.CIRCLE_CENTER)
            print("Math:", f"{new_tp[0][0].position[0]} - {orientation.get_orientation.CIRCLE_CENTER[0]} =", new_tp[0][0].position[0]- orientation.get_orientation.CIRCLE_CENTER[0], "\n",
                  f"{new_tp[0][0].position[1]} - {orientation.get_orientation.CIRCLE_CENTER[1]} =", new_tp[0][0].position[1]- orientation.get_orientation.CIRCLE_CENTER[1])
            new_tp[0][0].position[0]-=orientation.get_orientation.CIRCLE_CENTER[0]
            new_tp[0][0].position[1]-=orientation.get_orientation.CIRCLE_CENTER[1]
            print("New Position:", new_tp[0][0].position)
            print("\n---\n")

        segments.append(trajectory_pair_copy[0])

    # for trajectory_pair in data:
    #     tp = copy.deepcopy(trajectory_pair)
    #     # subtract circle center to 0 center the data
    #     tp[0][0].position[0]-= orientation.get_orientation.CIRCLE_CENTER[0]
    #     tp[0][0].position[1]-= orientation.get_orientation.CIRCLE_CENTER[1]
    #     tp[0][1].position[0]-= orientation.get_orientation.CIRCLE_CENTER[0]
    #     tp[0][1].position[1]-= orientation.get_orientation.CIRCLE_CENTER[1]
    #     tp[1][0].position[0]-= orientation.get_orientation.CIRCLE_CENTER[0]
    #     tp[1][0].position[1]-= orientation.get_orientation.CIRCLE_CENTER[1]
    #     tp[1][1].position[0]-= orientation.get_orientation.CIRCLE_CENTER[0]
    #     tp[1][1].position[1]-= orientation.get_orientation.CIRCLE_CENTER[1]

    #     if tp[0][0].position[0] // 25 < -30 or tp[0][0].position[1] // 25 < -30:
    #         new_tp = copy.deepcopy(trajectory_pair)
    #         print("Initial Position:", trajectory_pair[0][0].position, "\n", tp[0][0].position[0] // 25, "\n", tp[0][0].position[1] // 25)
    #         print("Post Centering Positions:", tp[0][0].position)
    #         print("CIRCLE_CENTER:", orientation.get_orientation.CIRCLE_CENTER)
    #         print("Math:", f"{new_tp[0][0].position[0]} - {orientation.get_orientation.CIRCLE_CENTER[0]} =", new_tp[0][0].position[0]- orientation.get_orientation.CIRCLE_CENTER[0], "\n",
    #               f"{new_tp[0][0].position[1]} - {orientation.get_orientation.CIRCLE_CENTER[1]} =", new_tp[0][0].position[1]- orientation.get_orientation.CIRCLE_CENTER[1])
    #         new_tp[0][0].position[0]-=orientation.get_orientation.CIRCLE_CENTER[0]
    #         new_tp[0][0].position[1]-=orientation.get_orientation.CIRCLE_CENTER[1]
    #         print("New Position:", new_tp[0][0].position)
    #         print("\n---\n")
    #     segments.append(tp[0])
    
    plot_xy_from_segments(segments, f"xy_distribution_post_centering.png")
    
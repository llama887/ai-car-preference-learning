import pickle

import subsample_state


def generate_grid_points():
    data = subsample_state.get_grid_points(samples=3000000)
    with open("grid_points.pkl", "wb") as f:
        pickle.dump(data, f)

    combined_data = []
    for data_set in data:
        combined_data.extend(data_set)
    with open("grid_points_segments.pkl", "wb") as f:
        pickle.dump(combined_data, f)

if __name__ == "__main__":
    generate_grid_points()

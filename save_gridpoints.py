import pickle

import subsample_state


def generate_grid_points():
    data = subsample_state.get_grid_points()
    with open("grid_points.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    generate_grid_points()

import pickle

import subsample_state

data = subsample_state.get_grid_points(2000000)
with open("grid_points.pkl", "wb") as f:
    pickle.dump(data, f)

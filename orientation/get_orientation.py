import pandas as pd
import numpy as np

# Precompute the angle dictionary
def precompute_angle_dict():
    df = pd.read_csv("orientation_data.csv")
    angle_dict = {}

    def _create_key(x_range, y_range):
        return (x_range[0], x_range[1], y_range[0], y_range[1])
    
    filter_conditions = [
        ((400, 600), (-np.inf, -600)),
        ((600, 800), (-np.inf, -600)),
        ((800, 1200), (-np.inf, -600)),
        ((1200, 1400), (-np.inf, -600)),
        ((1400, np.inf), (-np.inf, -600)),
        ((1400, np.inf), (-800, -600)),
        ((1400, np.inf), (-600, -400)),
        ((1400, np.inf), (-400, np.inf)),
        ((1200, 1400), (-400, np.inf)),
        ((600, 1200), (-400, np.inf)),
        ((0, 600), (-200, np.inf)),
        ((0, 600), (-400, -200)),
        ((0, 600), (-600, -400)),
        ((0, 400), (-np.inf, -600)),
        ((0, 600), (-np.inf, -600)),
    ]

    for x_range, y_range in filter_conditions:
        mask = (df['X'] >= x_range[0]) & (df['X'] < x_range[1]) & \
               (-df['Y'] > y_range[0]) & (-df['Y'] <= y_range[1])
        filtered_df = df[mask]
        if not filtered_df.empty:
            key = _create_key(x_range, y_range)
            angle_dict[key] = filtered_df["Angle"].mean()

    # Hard coded values for the bottom where too many cars die
    angle_dict[(600, 800, -np.inf, -600)] = 0
    angle_dict[(800, 1200, -np.inf, -600)] = 0

    return angle_dict

# Global variable to store the precomputed angle dictionary
ANGLE_DICT = precompute_angle_dict()

def get_angle(x, y):
    y = -y  # Invert y to match the original function's coordinate system

    for (x_min, x_max, y_min, y_max), angle in ANGLE_DICT.items():
        if x_min <= x < x_max and y_min < y <= y_max:
            return angle
    
    print(f"No data available for this section {x}, {y}")
    return None

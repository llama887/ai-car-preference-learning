import numpy as np
import yaml
import os

def get_circle_center():
    CENTER_FILE = "circle_center.yaml"
    if os.path.exists(CENTER_FILE):
        with open(CENTER_FILE) as f:
            data = yaml.safe_load(f)
            return data['cx'], data['cy']
    # Fallback logic
    return 1418.5, 1080.0

def get_angle(x, y):
    """Compute tangent-aligned angle (0° = upward) using circle geometry"""
    cx, cy = get_circle_center()
    return np.degrees(np.arctan2(y - cy, x - cx)) + 90.0

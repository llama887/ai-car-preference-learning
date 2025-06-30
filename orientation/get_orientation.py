import numpy as np

# Hardcoded circle center for tangent calculation (precomputed once)
CIRCLE_CX = 1418.5
CIRCLE_CY = 1080.0

def get_angle(x, y):
    """Compute tangent-aligned angle (0° = upward) using circle geometry"""
    return np.degrees(np.arctan2(y - CIRCLE_CY, x - CIRCLE_CX)) + 90.0

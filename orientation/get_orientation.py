import numpy as np
import cv2

# Precompute circle center on first import
CIRCLE_CENTER = None

# Compute center only once
def _compute_center():
    """Internal function to compute circle center from map"""
    try:
        img = cv2.imread("maps/map.png", cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), _ = cv2.minEnclosingCircle(largest_contour)
            print(f"Computed circle center: ({x:.2f}, {y:.2f})")
            return (float(x), float(y))
    except Exception as e:
        print(f"Circle center computation failed: {e}")
        

# Initialize center at import (only once)
if CIRCLE_CENTER is None:
    CIRCLE_CENTER = _compute_center()
    assert CIRCLE_CENTER is not None, "Failed to compute circle center from map image."

def get_angle(x, y):
    """Compute tangent-aligned angle (0° = upward) using track geometry"""
    return float(np.degrees(np.arctan2(y,x)) - 90.0)

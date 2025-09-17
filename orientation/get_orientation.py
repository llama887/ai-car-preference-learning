import numpy as np
import cv2

import csv
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans

DATA_PATH: Path = Path("orientation_data.csv")
MAX_CLUSTERS: int = 100
RANDOM_STATE: int = 42

# Globals initialized at import
_GLOBAL_KMEANS: KMeans | None = None
_GLOBAL_MEAN_ANGLES_DEG: np.ndarray | None = None
_GLOBAL_N_CLUSTERS: int = 0


def _load_xy_angles(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (X, Y, Angle) rows from CSV with header [X, Y, Angle].
    Returns:
        positions: shape (N, 2) float32
        angles_deg: shape (N,) float32 in [0, 360)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    xs: list[float] = []
    ys: list[float] = []
    angles: list[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or [h.strip().lower() for h in header] != ["x", "y", "angle"]:
            raise ValueError("CSV must have header exactly: X,Y,Angle")

        for row in reader:
            if len(row) < 3:
                continue
            x = float(row[0])
            y = float(row[1])
            a = float(row[2]) % 360.0
            xs.append(x)
            ys.append(y)
            angles.append(a)

    if not xs:
        raise ValueError("No data rows found in CSV.")

    positions = np.column_stack([np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)])
    angles_deg = np.array(angles, dtype=np.float32)
    return positions, angles_deg


def _fit_kmeans(positions: np.ndarray, desired_clusters: int) -> KMeans:
    """
    Fit KMeans to positions. If data points < desired_clusters, reduce cluster count.
    """
    n_samples = positions.shape[0]
    n_clusters = min(desired_clusters, n_samples)
    if n_clusters < 1:
        raise ValueError("Insufficient samples for clustering.")

    # n_init default varies by sklearn version; set explicitly for stability.
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(positions)
    return kmeans


def _circular_mean_degrees(degrees: np.ndarray) -> float:
    """
    Circular mean of angles in degrees, returned in [0, 360).
    """
    radians = np.deg2rad(degrees)
    mean_cos = np.mean(np.cos(radians))
    mean_sin = np.mean(np.sin(radians))
    mean_rad = math.atan2(mean_sin, mean_cos)
    mean_deg = math.degrees(mean_rad) % 360.0
    return mean_deg


def _compute_cluster_mean_angles(
    positions: np.ndarray, angles_deg: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """
    Compute circular mean angle per cluster.
    Returns:
        mean_angles_deg: shape (n_clusters,) with mean angle per cluster in [0, 360).
    """
    mean_angles: np.ndarray = np.zeros((n_clusters,), dtype=np.float32)
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            # Should not happen with KMeans, but guard anyway
            mean_angles[k] = 0.0
            continue
        mean_angles[k] = _circular_mean_degrees(angles_deg[mask])
    return mean_angles


def _initialize_models() -> None:
    """
    Load data, fit KMeans, compute per-bucket circular mean angles. Stores globals.
    """
    global _GLOBAL_KMEANS, _GLOBAL_MEAN_ANGLES_DEG, _GLOBAL_N_CLUSTERS

    positions, angles_deg = _load_xy_angles(DATA_PATH)
    kmeans = _fit_kmeans(positions, MAX_CLUSTERS)
    labels = kmeans.labels_
    n_clusters = kmeans.n_clusters

    mean_angles_deg = _compute_cluster_mean_angles(positions, angles_deg, labels, n_clusters)

    _GLOBAL_KMEANS = kmeans
    _GLOBAL_MEAN_ANGLES_DEG = mean_angles_deg
    _GLOBAL_N_CLUSTERS = n_clusters




# Initialize at import time
if _GLOBAL_KMEANS is None or _GLOBAL_MEAN_ANGLES_DEG is None:
    _initialize_models()


# Precompute circle center on first import
CIRCLE_CENTER = None

# Compute center only once
def _compute_center():
    """Internal function to compute circle center from map"""
    try:
        img = cv2.imread("maps/circle.png", cv2.IMREAD_GRAYSCALE)
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


def get_angle(x: float, y: float) -> float:
    """
    Given an (x, y) point, return the mean angle (degrees in [0, 360))
    of its nearest KMeans bucket.
    """
    if _GLOBAL_KMEANS is None or _GLOBAL_MEAN_ANGLES_DEG is None:
        raise RuntimeError("Model not initialized. Import this module or call _initialize_models() first.")
    label: int = int(_GLOBAL_KMEANS.predict(np.array([[x, y]], dtype=np.float32))[0])
    return float(_GLOBAL_MEAN_ANGLES_DEG[label])
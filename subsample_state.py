from orientation.get_orientation import get_angle
import argparse
import multiprocessing
import os
import pickle
import time

import numpy as np
import pygame
from imageio.v2 import imread
from shapely.geometry import Polygon
from shapely.vectorized import contains
from skimage import measure
from skimage.color import rgb2gray, rgba2rgb
from tqdm import tqdm

from agent import (
    CAR_SIZE_X,
    CAR_SIZE_Y,
    HEIGHT,
    WIDTH,
    Car,
    StateActionPair,
    segment_list_to_dict,
)
from rules import check_rules_one

os.environ["SDL_VIDEODRIVER"] = "dummy"

number_of_rules = 3


def parallel_subsample_state(image_path,
                             number_of_points=100000,
                             epsilon=1e-4,
                             min_error=10,
                             grid_factor=10):

    def binary_search(lb, ub, target):
        # 1) Prepare candidates
        n_candidates = multiprocessing.cpu_count() * grid_factor
        resolutions = np.linspace(lb, ub, n_candidates)

        # 2) Parallel subsample
        params = [(image_path, float(r)) for r in resolutions]
        with multiprocessing.Pool() as pool:
            subsamples = pool.starmap(subsample_state, params)

        # 3) Count points
        counts = np.array([len(s) for s in subsamples])

        # 4) Tolerance
        err = max(target * epsilon, min_error)

        # 5) Exact match
        for s, count in zip(subsamples, counts):
            if abs(count - target) <= err:
                return s

        # 6) Bracket target
        #    We assume counts decreases monotonically with resolution.
        #    Find i such that counts[i-1] > target > counts[i]
        idx = np.where(counts < target)[0]
        if idx.size == 0 or idx[0] == 0:
            raise ValueError("Target out of range at current bounds.")

        i = idx[0]
        # resolutions sorted ascending,
        # counts[i-1] > target > counts[i]
        lo_res = resolutions[i-1]
        hi_res = resolutions[i]
        lo_count = counts[i-1]
        hi_count = counts[i]

        # 7) Endpoint tolerance check
        if abs(lo_count - target) <= err:
            return subsamples[i-1]
        if abs(hi_count - target) <= err:
            return subsamples[i]

        # 8) Recurse with correct order: lo_res < hi_res
        return binary_search(lo_res, hi_res, target)
    try:
        return binary_search(6.5, 7.5, number_of_points) # tuners 
    except ValueError as e:
        print(
            f"Error: {e}. Re-attempting with different resolution range." 
        )
        return binary_search(5.5, 6.5, number_of_points)


def subsample_state(image_path, grid_resolution, tolerance=1.0):
    # Read the image
    polypic = imread(image_path)

    # Check if the image has an alpha channel and convert only if necessary
    if polypic.shape[-1] == 4:  # RGBA
        polypic = rgba2rgb(polypic)

    # Convert to grayscale
    gray = rgb2gray(polypic)

    # Detect contours in the grayscale image at an appropriate level
    contours = measure.find_contours(gray, 0.5)

    if len(contours) < 2:
        print(
            "Not enough contours found. Please check the contour level or image content."
        )
        return []

    # Sort contours by length (assuming largest is the outer boundary, second largest is inner boundary)
    contours = sorted(contours, key=len, reverse=True)
    outer_polygon = Polygon(contours[0]).simplify(tolerance)
    inner_polygon = Polygon(contours[1]).simplify(tolerance)

    # Create a ring-shaped polygon by subtracting the inner polygon from the outer polygon
    track_polygon = outer_polygon.difference(inner_polygon)

    # Get the bounding box for the polygon
    min_x, min_y, max_x, max_y = track_polygon.bounds

    # Generate grid points and filter them in bulk
    x, y = np.meshgrid(
        np.arange(min_x, max_x, grid_resolution),
        np.arange(min_y, max_y, grid_resolution),
    )
    points = np.vstack([x.ravel(), y.ravel()]).T
    valid_points = points[contains(track_polygon, points[:, 0], points[:, 1])]
    # print("Found", len(valid_points), "points.")
    return valid_points



def process_trajectory_segment(params):
    point_thing = None
    position_thing = None
    """Process a single trajectory segment for a given point, angle, and speed."""
    (
        point,
        angle_deviation,
        speed,
        CAR_SIZE_X,
        CAR_SIZE_Y,
        WIDTH,
        HEIGHT,
        game_map_path,
    ) = params

    trajectory_segments = []

    # Initialize car object
    car = Car()

    # Load game map
    game_map = pygame.image.load(game_map_path).convert()

    # Generate trajectory segments
    for first_action in range(0, 4):
        car.position = [point[1] - CAR_SIZE_X / 2, point[0] - CAR_SIZE_Y / 2]
        car_x = car.position[0]
        car_y = car.position[1]

        angle = angle_deviation + get_angle(car_x, car_y)
        car.speed = speed
        car.angle = angle
        car.radars.clear()
        car.rotated_sprite = car.rotate_center(car.sprite, car.angle)

        # Precompute cos/sin for angles
        cos_angle = np.cos(np.radians(360 - car.angle))
        sin_angle = np.sin(np.radians(360 - car.angle))

        car.check_radar(car.angle, game_map)
        first_state_action_pair = StateActionPair(
            [car.check_radar(d, game_map) for d in range(-90, 120, 45)],
            first_action,
            car.position,
            True,
        )
        point_thing = tuple(point.tolist())
        position_thing = tuple(car.position)

        # Apply action
        if first_action == 0:
            car.angle += 10  # Left
        elif first_action == 1:
            car.angle -= 10  # Right
        elif first_action == 2:
            if car.speed - 2 >= 12:
                car.speed -= 2  # Slow Down
        else:
            car.speed += 2  # Speed Up

        dx = cos_angle * car.speed
        dy = sin_angle * car.speed
        car.position += np.array([dx, dy])
        car.position[0] = np.clip(car.position[0], 20, WIDTH - 120)
        car.position[1] = np.clip(car.position[1], 20, HEIGHT - 120)
        for second_action in range(0, 4):
            last_state_action_pair = StateActionPair(
                [car.check_radar(d, game_map) for d in range(-90, 120, 45)],
                second_action,
                car.position,
                True,
            )
            trajectory_segments.append(
                [first_state_action_pair, last_state_action_pair]
            )

    return trajectory_segments, point_thing, position_thing


def split_by_rules(trajectory_segments):
    mini_gargantuar = [[], [], [], []]
    for segment in trajectory_segments:
        rule_counts, _, _ = check_rules_one(segment, number_of_rules)
        mini_gargantuar[rule_counts].append(segment)
    return mini_gargantuar


def get_grid_points(samples=2000000):
    # Load subsampled grid points
    ANGLE_STEP = 10
    MAX_ANGLE_DEVIATION = 10
    ANGLES = len(range(-MAX_ANGLE_DEVIATION, MAX_ANGLE_DEVIATION + 1, ANGLE_STEP))
    SPEED_STEP = 10
    SPEEDS = 40 // SPEED_STEP
    FIRST_ACTIONS = 4
    SECOND_ACTIONS = 4
    gridpoints = samples // ANGLES // SPEEDS // FIRST_ACTIONS // SECOND_ACTIONS
    if gridpoints < 1000:
        raise ValueError(f"{gridpoints} is not enough points to subsample.")

    print(f"Searching for {gridpoints} grid points")
    points = parallel_subsample_state("maps/map.png", gridpoints)
    print(f"Found {len(points)} points.")
    _ = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)

    # Prepare parameters for multiprocessing
    params = [
        (
            point,
            angle_deviation,
            speed,
            CAR_SIZE_X,
            CAR_SIZE_Y,
            WIDTH,
            HEIGHT,
            "maps/map.png",
        )
        for point in points
        for angle_deviation in range(
            -MAX_ANGLE_DEVIATION, MAX_ANGLE_DEVIATION + 1, ANGLE_STEP
        )
        for speed in range(10, 50, SPEED_STEP)
    ]

    print("Starting segment subsampling...")
    # Use multiprocessing to process trajectory segments
    with multiprocessing.Pool() as pool:
        tmp_results = list(
            tqdm(
                pool.imap_unordered(process_trajectory_segment, params),
                total=len(params),
            )
        )
    results = []
    point_set = set()
    position_set = set()
    for result in tmp_results:
        r, point, position = result
        results.append(r)
        point_set.add(point)
        position_set.add(position)

    return results


if __name__ == "__main__":
    start = time.time()
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )
    parse.add_argument(
        "-s",
        "--samples",
        type=int,
        nargs=1,
        help="Number of samples",
    )
    parse.add_argument(
        "-r",
        "--rules",
        type=int,
        nargs=1,
        help="Number of rules",
    )
    args = parse.parse_args()

    if args.rules:
        number_of_rules = args.rules[0]

    if args.samples and args.samples[0] > 0:
        samples = args.samples[0]
    else:
        samples = 2000000

    results = get_grid_points(samples)

    list_of_segments = []
    for segment in results:
        list_of_segments.extend(segment)
    print(f"Found {len(list_of_segments)} segments.")

    
    print(f"Saving to subsampled_gargantuar_1_length.pkl...")
    with open(f"subsampled_gargantuar_1_length.pkl", "wb") as f:
        pickle.dump(segment_list_to_dict(list_of_segments), f)
    end = time.time()
    print(f"Finished in {end - start} seconds.")


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
)
from rules import check_rules_one

os.environ["SDL_VIDEODRIVER"] = "dummy"

number_of_rules = 3


def parallel_subsample_state(image_path, number_of_points=100000, epsilon=0.0001):
    def binary_search(lower_bound, upper_bound, target):
        grid_resolutions = np.linspace(
            lower_bound, upper_bound, multiprocessing.cpu_count() * 10
        )
        params = [
            (image_path, float(grid_resolution)) for grid_resolution in grid_resolutions
        ]
        with multiprocessing.Pool() as pool:
            results = pool.starmap(subsample_state, params)
        points_numbers = np.array([len(result) for result in results])
        error_margin = target * epsilon
        if error_margin < 10:
            error_margin = 10
        index = None
        for i, num in enumerate(points_numbers):
            if target - error_margin <= num <= target + error_margin:
                return results[i]
            elif num < target:
                index = i
                closest_resolution_above = grid_resolutions[i - 1]
                closest_resolution_below = grid_resolutions[i]
                closest_n_points_above = points_numbers[i - 1]
                closest_n_points_below = points_numbers[i]
                break
        if (
            index
            and closest_n_points_above
            and closest_n_points_below
            and closest_resolution_above
            and closest_resolution_below
        ):
            if abs(closest_n_points_above - target) < target * epsilon:
                # print(f"above: {results[index]}")
                return results[index]
            if abs(closest_resolution_below - target) < target * epsilon:
                # print(f"below: {results[index]}")
                return results[index]
            # print(f"above: {results[index]}, below: {results[index]}")
            return binary_search(
                closest_resolution_below, closest_resolution_above, target
            )
        else:
            raise ValueError("Target out of range for the given resolutions.")

    # Initial bounds
    lower_bound = 6.5
    upper_bound = 7.5
    return binary_search(lower_bound, upper_bound, number_of_points)


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


def get_grid_points(samples=2000000, number_of_rules=1):
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

    print("Splitting by rules...")
    with multiprocessing.Pool() as pool:
        split_results = list(
            tqdm(pool.imap_unordered(split_by_rules, results), total=len(results))
        )

    gargantuar = [[], [], [], []]
    for result in split_results:
        for i in range(len(result)):
            gargantuar[i].extend(result[i])

    for i in range(len(gargantuar)):
        print(f"{len(gargantuar[i])} segments of {i} rules followed")
    total_segments = sum([len(gargantuar[i]) for i in range(len(gargantuar))])
    print(f"{total_segments} total segments")
    print(f"Saving to subsampled_gargantuar_1_length_{number_of_rules}_rules.pkl...")
    with open(f"subsampled_gargantuar_1_length_{number_of_rules}_rules.pkl", "wb") as f:
        pickle.dump(gargantuar, f)
    end = time.time()
    print(f"Finished in {end - start} seconds.")

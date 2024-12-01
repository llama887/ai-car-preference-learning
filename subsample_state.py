import argparse
import math
import multiprocessing
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pygame
from imageio.v2 import imread
from shapely.geometry import Point, Polygon
from skimage import measure
from skimage.color import rgb2gray, rgba2rgb

from agent import CAR_SIZE_X, CAR_SIZE_Y, HEIGHT, WIDTH, Car, StateActionPair


def generate_points(track_polygon, resolution):
    # Generate grid points for a given resolution
    min_x, min_y, max_x, max_y = track_polygon.bounds
    points = [
        (x, y)
        for x in np.arange(min_x, max_x, resolution)
        for y in np.arange(min_y, max_y, resolution)
        if track_polygon.contains(Point(x, y))
    ]
    return len(points), resolution, points


def binary_search_parallel(
    track_polygon, target_points, low, high, tolerance=1000, num_workers=None
):
    if abs(low - high) < 1 and tolerance < float("inf"):
        # Stop if resolution difference is negligible and within tolerance
        return None, []

    if num_workers is None:
        print("Using all available cores:", os.cpu_count())
        num_workers = os.cpu_count()

    mid1 = (2 * low + high) / 3
    mid2 = (low + 2 * high) / 3
    resolutions = [low, mid1, mid2, high]

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(
            generate_points, [(track_polygon, res) for res in resolutions]
        )

    closest_diff = float("inf")
    best_resolution, best_points = None, []
    closest_num_points = None

    for num_points, resolution, points in results:
        diff = abs(num_points - target_points)
        if diff < closest_diff:
            closest_diff = diff
            closest_num_points = num_points
            best_resolution = resolution
            best_points = points

    # Print the progress for user feedback
    print(
        f"Current search range: [{low:.2f}, {high:.2f}]. "
        f"Closest points so far: {closest_num_points} (target: {target_points}). "
        f"Best resolution: {best_resolution:.2f}"
    )

    # Check if we are within the tolerance for the closest resolution
    if closest_diff <= tolerance:
        return best_resolution, best_points

    # Recursively narrow down the range
    if best_resolution == low or best_resolution == high:
        return best_resolution, best_points
    elif best_resolution == mid1:
        return binary_search_parallel(
            track_polygon, target_points, low, mid2, tolerance, num_workers
        )
    else:  # best_resolution == mid2
        return binary_search_parallel(
            track_polygon, target_points, mid1, high, tolerance, num_workers
        )


def subsample_state(
    image_path, number_of_points=1000, num_workers=None, tolerance=1000
):
    polypic = imread(image_path)

    if polypic.shape[-1] == 4:  # RGBA
        polypic = rgba2rgb(polypic)

    gray = rgb2gray(polypic)
    contours = measure.find_contours(gray, 0.5)

    if len(contours) < 2:
        print("Not enough contours found. Check image content or contour level.")
        return []

    contours = sorted(contours, key=len, reverse=True)
    outer_contour = contours[0]
    inner_contour = contours[1]

    outer_polygon = Polygon(outer_contour).simplify(1.0)
    inner_polygon = Polygon(inner_contour).simplify(1.0)
    track_polygon = outer_polygon.difference(inner_polygon)

    low, high = 0.01, 1000.0
    optimal_resolution, best_points = binary_search_parallel(
        track_polygon, number_of_points, low, high, tolerance, num_workers
    )

    print(f"Optimal grid resolution: {optimal_resolution}")
    return best_points


def process_trajectory_segment(params):
    point, angle, speed, game_map_path = params
    trajectory_segments = []

    car = Car()
    car.position = [point[1] - CAR_SIZE_X / 2, point[0] - CAR_SIZE_Y / 2]
    car.speed = speed
    car.angle = angle
    car.radars.clear()
    car.rotated_sprite = car.rotate_center(car.sprite, car.angle)
    game_map = pygame.image.load(game_map_path).convert()

    radar_results = [
        car.check_radar(d, game_map) for d in range(-90, 120, 45)
    ]  # Cache radar results

    for first_action in range(4):
        car.radars = radar_results
        first_state_action_pair = StateActionPair(
            radar_results, first_action, car.position, True
        )

        # Apply action
        if first_action == 0:
            car.angle += 10
        elif first_action == 1:
            car.angle -= 10
        elif first_action == 2:
            car.speed = max(car.speed - 2, 12)
        else:
            car.speed += 2

        car.position[0] += math.cos(math.radians(360 - car.angle)) * car.speed
        car.position[0] = max(20, min(car.position[0], WIDTH - 120))
        car.position[1] += math.sin(math.radians(360 - car.angle)) * car.speed
        car.position[1] = max(20, min(car.position[1], HEIGHT - 120))

        radar_results = [car.check_radar(d, game_map) for d in range(-90, 120, 45)]
        for second_action in range(4):
            last_state_action_pair = StateActionPair(
                radar_results, second_action, car.position, True
            )
            trajectory_segments.append(
                [first_state_action_pair, last_state_action_pair]
            )

    return trajectory_segments


if __name__ == "__main__":
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

    args = parse.parse_args()
    if args.samples and args.samples[0] > 0:
        samples = args.samples[0]
    else:
        samples = 1000

    # Load subsampled grid points
    gridpoints = samples // (360 / 10) // (40 / 2) // 4 // 4
    print("Gridding the map")
    points = subsample_state("maps/map.png", gridpoints)

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
    game_map = pygame.image.load("maps/map.png").convert()

    # Prepare parameters for multiprocessing
    params = [
        (point, angle, speed, CAR_SIZE_X, CAR_SIZE_Y, WIDTH, HEIGHT, "maps/map.png")
        for point in points
        for angle in range(0, 360, 10)
        for speed in range(10, 50, 2)
    ]

    # Use multiprocessing to process trajectory segments
    print("Gridding over all actions, speeds, and orientations")
    with multiprocessing.Pool() as pool:
        results = pool.map(process_trajectory_segment, params)

    trajectory_segments = [[result[0], result[-1]] for result in results if result]
    assert isinstance(
        trajectory_segments[0][0], StateActionPair
    ), "Error in subsampling"
    assert isinstance(
        trajectory_segments[-1][-1], StateActionPair
    ), "Error in subsampling"

    with open("subsampled_gargantuan.plk", "wb") as f:
        pickle.dump(trajectory_segments, f)

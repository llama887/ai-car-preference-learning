import argparse
import math
import multiprocessing
import pickle

import numpy as np
import pygame
from imageio.v2 import imread
from shapely.geometry import Point, Polygon
from skimage import measure
from skimage.color import rgb2gray, rgba2rgb

from agent import CAR_SIZE_X, CAR_SIZE_Y, HEIGHT, WIDTH, Car, StateActionPair


def subsample_state(image_path, number_of_points=1000):
    # Read the image
    image_path = "maps/map.png"  # Replace with the correct path if necessary
    polypic = imread(image_path)

    # Check if the image has an alpha channel and convert only if necessary
    if polypic.shape[-1] == 4:  # RGBA
        polypic = rgba2rgb(polypic)

    # Convert to grayscale
    gray = rgb2gray(polypic)

    # Detect contours in the grayscale image at an appropriate level
    contours = measure.find_contours(gray, 0.5)

    # Check if contours were found
    if len(contours) < 2:
        print(
            "Not enough contours found. Please check the contour level or image content."
        )
        return []
    else:
        # Sort contours by length (assuming largest is the outer boundary, second largest is inner boundary)
        contours = sorted(contours, key=len, reverse=True)
        outer_contour = contours[0]
        inner_contour = contours[1]

        # Create polygons from the contours without swapping coordinates
        outer_polygon = Polygon(outer_contour).simplify(1.0)
        inner_polygon = Polygon(inner_contour).simplify(1.0)

        # Create a ring-shaped polygon by subtracting the inner polygon from the outer polygon
        track_polygon = outer_polygon.difference(inner_polygon)

        # Print polygon bounds for debugging
        print("Track polygon bounds:", track_polygon.bounds)

        # Define grid resolution (try a smaller value if points are sparse)
        grid_resolution = 100  # Adjust this value as needed

        # Get the bounding box for the polygon
        min_x, min_y, max_x, max_y = track_polygon.bounds

        # Generate grid points within the bounding box and filter those inside the track polygon
        points = []
        for x in np.arange(min_x, max_x, grid_resolution):
            for y in np.arange(min_y, max_y, grid_resolution):
                point = Point(x, y)
                if track_polygon.contains(point):
                    points.append((x, y))

        return points, contours


def process_trajectory_segment(params):
    """Process a single trajectory segment for a given point, angle, and speed."""
    point, angle, speed, CAR_SIZE_X, CAR_SIZE_Y, WIDTH, HEIGHT, game_map_path = params
    trajectory_segments = []

    # Initialize car object
    car = Car()
    car.position = [point[1] - CAR_SIZE_X / 2, point[0] - CAR_SIZE_Y / 2]
    car.speed = speed
    car.angle = angle
    car.radars.clear()
    car.rotated_sprite = car.rotate_center(car.sprite, car.angle)

    # Load game map
    game_map = pygame.image.load(game_map_path).convert()

    # Generate trajectory segments
    for first_action in range(0, 4):
        car.check_radar(car.angle, game_map)
        first_state_action_pair = StateActionPair(
            [car.check_radar(d, game_map) for d in range(-90, 120, 45)],
            first_action,
            car.position,
            True,
        )

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

        car.position[0] += math.cos(math.radians(360 - car.angle)) * car.speed
        car.position[0] = max(car.position[0], 20)
        car.position[0] = min(car.position[0], WIDTH - 120)
        car.position[1] += math.sin(math.radians(360 - car.angle)) * car.speed
        car.position[1] = max(car.position[1], 20)
        car.position[1] = min(car.position[1], HEIGHT - 120)

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

    # Load subsampled grid points and contours
    gridpoints = samples // (360 / 10) // (40 / 2) // 4 // 4
    points, contours = subsample_state("maps/map.png", gridpoints)

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
    game_map = pygame.image.load("maps/map.png").convert()

    points, _ = subsample_state("maps/map.png", args.samples)

    # Prepare parameters for multiprocessing
    params = [
        (point, angle, speed, CAR_SIZE_X, CAR_SIZE_Y, WIDTH, HEIGHT, "maps/map.png")
        for point in points
        for angle in range(0, 360, 10)
        for speed in range(10, 50, 2)
    ]

    # Use multiprocessing to process trajectory segments
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

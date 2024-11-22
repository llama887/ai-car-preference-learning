import math
import time

import numpy as np
import pygame
from imageio.v2 import imread
from shapely.geometry import Point, Polygon
from skimage import measure
from skimage.color import rgb2gray, rgba2rgb

from agent import CAR_SIZE_X, CAR_SIZE_Y, HEIGHT, WIDTH, Car


def subsample_state(image_path, number_of_points):
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
        grid_resolution = 10  # Adjust this value as needed

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


def calculate_tangent_angle(position, contours):
    # Flatten all contour points into one list
    all_points = np.vstack(contours)

    # Find the closest point on the contour to the given position
    distances = np.linalg.norm(all_points - np.array(position), axis=1)
    closest_index = np.argmin(distances)
    closest_point = all_points[closest_index]

    # Determine the tangent vector
    if closest_index > 0:
        prev_point = all_points[closest_index - 1]
    else:
        prev_point = all_points[closest_index]

    if closest_index < len(all_points) - 1:
        next_point = all_points[closest_index + 1]
    else:
        next_point = all_points[closest_index]

    tangent_vector = next_point - prev_point

    # Calculate the tangent angle
    angle = math.degrees(math.atan2(tangent_vector[1], tangent_vector[0]))
    return angle


# Load subsampled grid points and contours
points, contours = subsample_state("maps/map.png", 100)

# Initialize pygame and load the map
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)
screen.blit(pygame.image.load("maps/map.png").convert(), (0, 0))

# Create a Car object with initial position and orientation
while True:
    random_point = points[np.random.choice(len(points))]
    car = Car()
    car.position = [random_point[1] - CAR_SIZE_X / 2, random_point[0] - CAR_SIZE_Y / 2]

    # Calculate the tangent angle at the random point
    car.angle = calculate_tangent_angle(car.position, contours)

    # Draw the car
    car.draw(screen)
    pygame.display.flip()
    time.sleep(0.1)

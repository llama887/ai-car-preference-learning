import matplotlib.pyplot as plt
import numpy as np
from imageio.v2 import imread  # Explicitly use imageio.v2 to avoid deprecation warning
from shapely.geometry import Point, Polygon
from skimage import measure
from skimage.color import rgb2gray

# Read the image
image_path = "maps/map.png"  # Replace with the correct path if necessary
polypic = imread(image_path)

# Check if the image has an alpha channel and convert only if necessary
if polypic.shape[-1] == 4:  # RGBA
    from skimage.color import rgba2rgb

    polypic = rgba2rgb(polypic)

# Convert to grayscale
gray = rgb2gray(polypic)

# Detect contours in the grayscale image at an appropriate level
contours = measure.find_contours(gray, 0.5)


# Check if contours were found
if len(contours) < 2:
    print("Not enough contours found. Please check the contour level or image content.")
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

    # Check if points were generated
    if points:
        x_coords, y_coords = zip(*points)  # Unpack as (x, y) for plotting
    else:
        print("No points found within the track polygon.")

    # Plot the results
    plt.imshow(
        gray, cmap="gray", origin="upper", extent=[0, gray.shape[1], gray.shape[0], 0]
    )  # Use extent to align coordinates with image dimensions
    plt.plot(
        *zip(*outer_polygon.exterior.coords), color="red"
    )  # Outline of the outer polygon
    plt.plot(
        *zip(*inner_polygon.exterior.coords), color="green"
    )  # Outline of the inner polygon
    if points:
        plt.scatter(
            x_coords, y_coords, s=1, color="blue", alpha=0.6
        )  # Plot with (x, y) orientation
    plt.axis("equal")
    plt.show()

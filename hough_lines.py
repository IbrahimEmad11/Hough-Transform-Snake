import numpy as np
import matplotlib.pyplot as plt
from canny_detection import CannyEdgeDetector
import cv2
from PIL import Image

def hough_line_transform(image_edges, theta_resolution=1, rho_resolution=1):
    # Define the Hough space
    height, width = image_edges.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    num_thetas = int(np.degrees(np.pi) / theta_resolution)
    accumulator = np.zeros((2 * max_rho, num_thetas), dtype=np.uint8)

    # Define thetas
    thetas = np.linspace(-np.pi/2, np.pi/2, num_thetas)

    # Loop over edge pixels
    edge_points = np.nonzero(image_edges)
    for i in range(len(edge_points[0])):
        x = edge_points[1][i]
        y = edge_points[0][i]
        for theta_index, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            accumulator[rho + max_rho, theta_index] += 1

    return accumulator, thetas, np.arange(-max_rho, max_rho, rho_resolution)

def find_hough_peaks(accumulator, thetas, rhos, threshold):
    peaks = []
    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i, j] > threshold:
                peaks.append((i, j))
    return peaks

def draw_lines(image, peaks, thetas, rhos):
    for peak in peaks:
        rho_index, theta_index = peak
        rho = rhos[rho_index]
        theta = thetas[theta_index]
        x0 = rho * np.cos(theta)
        y0 = rho * np.sin(theta)
        x1 = int(x0 + 1000 * (-np.sin(theta)))
        y1 = int(y0 + 1000 * (np.cos(theta)))
        x2 = int(x0 - 1000 * (-np.sin(theta)))
        y2 = int(y0 - 1000 * (np.cos(theta)))
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

def detect_and_draw_hough_lines(image_edges,image, theta_resolution=1, rho_resolution=1, threshold=200):
    # canny_detector = CannyEdgeDetector(image)
    # image_edges = canny_detector.detect_edges()
    accumulator, thetas, rhos = hough_line_transform(image_edges, theta_resolution, rho_resolution)
    peaks = find_hough_peaks(accumulator, thetas, rhos, threshold)
    image_with_lines = draw_lines(image, peaks, thetas, rhos)
    return image_with_lines

###################################################################### HOUGH CIRCLES ####################################################################

def _hough_circles(image_path, min_radius=20, max_radius=100, param1=50, param2=30):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply median blur
    image = cv2.medianBlur(image, 5)

    # Define accumulator
    accumulator = np.zeros_like(image, dtype=np.uint16)

    # Define radius range
    radius_range = np.arange(min_radius, max_radius + 1)

    # Precompute cosine and sine values for all thetas
    cos_theta = np.cos(np.deg2rad(np.arange(360)))
    sin_theta = np.sin(np.deg2rad(np.arange(360)))

    # Loop over radius
    for radius in radius_range:
        for theta_index in range(360):
            # Compute circle parameters
            a = np.round(radius * cos_theta[theta_index]).astype(int)
            b = np.round(radius * sin_theta[theta_index]).astype(int)

            # Shifted coordinates
            center_x = np.arange(max(0, a), min(image.shape[1], image.shape[1] - a))
            center_y = np.arange(max(0, b), min(image.shape[0], image.shape[0] - b))

            # Create meshgrid for indexing
            grid_x, grid_y = np.meshgrid(center_x, center_y)

            # Mask to extract edge points
            edge_mask = image[grid_y, grid_x] > 0

            # Update accumulator using vectorized operations
            accumulator[grid_y[edge_mask], grid_x[edge_mask]] += 1

    # Threshold the accumulator to find potential circles
    circles = np.argwhere(accumulator >= param2)

    # Check if circles were detected
    if circles.size == 0:
        print("No circles detected")
        return None

    # Convert circles to integer coordinates
    circles = np.uint16(np.around(circles))

    # Convert image to BGR for drawing circles
    cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw detected circles
    if circles is not None:
        for circle in circles:
            center = (circle[1], circle[0])
            radius = circle[2]  # Make sure to update this line if the structure of circles array changes
            cv2.circle(cimg, center, radius, (0, 255, 0), 2)
            cv2.circle(cimg, center, 2, (0, 0, 255), 3)

    # Convert the resulting image to a PIL Image
    result_image = Image.fromarray(cimg)

    return result_image


def hough_circles(image_path, min_radius=20, max_radius=100, dp=1, min_dist=50, param1=50, param2=30):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply median blur
    image = cv2.medianBlur(image, 5)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)

    print("Hough is: ", circles)
    if circles is None:
      print("No Circles")

    # Convert circles to integer coordinates
    circles = np.uint16(np.around(circles))

    # Convert image to BGR for drawing circles
    cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw detected circles
    if circles is not None:
        for circle in circles[0]:
            center = (circle[0], circle[1])
            radius = circle[2]
            cv2.circle(cimg, center, radius, (0, 255, 0), 2)
            cv2.circle(cimg, center, 2, (0, 0, 255), 3)

    # Convert the resulting image to a PIL Image
    result_image = Image.fromarray(cimg)

    return result_image


import numpy as np
import matplotlib.pyplot as plt
from canny_detection import CannyEdgeDetector
import cv2

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

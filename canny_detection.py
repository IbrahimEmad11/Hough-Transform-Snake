import numpy as np
import cv2
from PIL import Image
import math
import matplotlib.pyplot as plt

class CannyEdgeDetector:
    def __init__(self, image):
        self.image = image
        self.edges = None

    def grayscale(self, image):
        if isinstance(image, np.ndarray):
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif isinstance(image, Image.Image):
            return np.array(image.convert('L'))
        else:
            raise ValueError("Unsupported image format. Supported formats: NumPy array, PIL Image")

    def gaussian_blur(self, image, kernel_size=5, sigma=1):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def convolve(self, image, kernel):
        if image is not None:
            height, width = image.shape
            k_height, k_width = kernel.shape
            output = np.zeros_like(image)
            padded_image = np.pad(image, ((k_height//2, k_height//2), (k_width//2, k_width//2)), mode='constant')

            for i in range(height):
                for j in range(width):
                    output[i, j] = np.sum(padded_image[i:i+k_height, j:j+k_width] * kernel)
            return output
         
    def sobel_kernels(self, image):
        if image is not None:
          
            image = image.astype(np.float32)

            # Sobel filter kernels
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

            # Convolve image with Sobel kernels to calculate gradients
            grad_x = self.convolve(image, kernel_x)
            grad_y = self.convolve(image, kernel_y)

            # Compute gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
            gradient_magnitude.astype(np.uint8)
            direction = np.arctan2(grad_x, grad_y)

            return gradient_magnitude, direction

    def non_max_suppression(self, gradient_matrix, theta_matrix):
        m, n = gradient_matrix.shape
        Z = np.zeros((m, n), dtype=np.uint8)
        angle = theta_matrix * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                q, r = 255, 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_matrix[i, j+1]
                    r = gradient_matrix[i, j-1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_matrix[i+1, j-1]
                    r = gradient_matrix[i-1, j+1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_matrix[i+1, j]
                    r = gradient_matrix[i-1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_matrix[i-1, j-1]
                    r = gradient_matrix[i+1, j+1]

                if (gradient_matrix[i, j] >= q) and (gradient_matrix[i, j] >= r):
                    Z[i, j] = gradient_matrix[i, j]
                else:
                    Z[i, j] = 0

        return Z

    def double_threshold(self, img_grayscale, low_TH_ratio=0.05, high_TH_ratio=0.09):
        high_TH = img_grayscale.max() * high_TH_ratio
        low_TH = high_TH * low_TH_ratio
        weak_value = 25
        strong_value = 255
        threshold_matrix = np.zeros_like(img_grayscale, dtype=np.uint8)

        strong_i, strong_j = np.where(img_grayscale >= high_TH)
        weak_i, weak_j = np.where((img_grayscale <= high_TH) & (img_grayscale >= low_TH))

        threshold_matrix[strong_i, strong_j] = strong_value
        threshold_matrix[weak_i, weak_j] = weak_value

        return threshold_matrix, weak_value, strong_value

    def hysteresis(self, img_grayscale, weak_value, strong_value=255):
        M, N = img_grayscale.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if img_grayscale[i, j] == weak_value:
                    try:
                        if (img_grayscale[i+1, j-1] == strong_value) or \
                           (img_grayscale[i+1, j] == strong_value) or \
                           (img_grayscale[i+1, j+1] == strong_value) or \
                           (img_grayscale[i, j-1] == strong_value) or \
                           (img_grayscale[i, j+1] == strong_value) or \
                           (img_grayscale[i-1, j-1] == strong_value) or \
                           (img_grayscale[i-1, j] == strong_value) or \
                           (img_grayscale[i-1, j+1] == strong_value):
                            img_grayscale[i, j] = strong_value
                        else:
                            img_grayscale[i, j] = 0
                    except IndexError:
                        pass
        return img_grayscale

    def detect_edges(self, low_TH_ratio=0.05, high_TH_ratio=0.09, kernel_size=5, sigma=1):
        gray_image = self.grayscale(self.image)
        blurred_image = self.gaussian_blur(gray_image, kernel_size, sigma)
        gradient_magnitude, gradient_direction = self.sobel_kernels(blurred_image)
        suppressed = self.non_max_suppression(gradient_magnitude, gradient_direction)
        threshold_matrix, weak_value, strong_value = self.double_threshold(suppressed, low_TH_ratio, high_TH_ratio)
        edges = self.hysteresis(threshold_matrix, weak_value, strong_value)
        self.edges = edges
        return self.edges



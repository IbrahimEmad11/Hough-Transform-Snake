import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QSlider , QColorDialog, QAction, QTextEdit, QMessageBox
from PyQt5.QtCore import QTimer,Qt, QPointF
from PyQt5.QtGui import QColor, QIcon, QCursor, QKeySequence, QPixmap, QImage, QPainter
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QProgressBar, QDialog, QVBoxLayout, QLineEdit, QLabel
from task2 import Ui_MainWindow
from PIL import Image , ImageFilter
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from canny_detection import CannyEdgeDetector
from hough_lines import *

# pyuic5 task1.ui -o task1.py
# global threshold - Normalize 
class SaltPepperDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Salt and Pepper: ")
        layout = QVBoxLayout()
        label1 = QLabel("Noise Ratio:")
        self.input1 = QLineEdit()

        layout.addWidget(label1)
        layout.addWidget(self.input1)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_action)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)
    
    def get_input_values(self):
        return self.input1.text()

    def apply_action(self):
         self.accept()

class ThresDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Global Threshold Parameter (0-255): ")
        layout = QVBoxLayout()
        label1 = QLabel("Value:")
        self.input1 = QLineEdit()

        layout.addWidget(label1)
        layout.addWidget(self.input1)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_action)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)
    
    def get_input_values(self):
        return self.input1.text()

    def apply_action(self):
         self.accept()

class UniformNoiseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Uniform Noise: ")
        layout = QVBoxLayout()
        label1 = QLabel("Noise Intensity:")
        self.input1 = QLineEdit()

        layout.addWidget(label1)
        layout.addWidget(self.input1)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_action)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def get_input_values(self):
        return self.input1.text()

    def apply_action(self):
         self.accept()

class GaussianNoiseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaussian Noise Parameters")
        layout = QVBoxLayout()
        label1 = QLabel("Mean:")
        label2 = QLabel("Std:")
        self.input1 = QLineEdit()
        self.input2 = QLineEdit()

        layout.addWidget(label1)
        layout.addWidget(self.input1)
        layout.addWidget(label2)
        layout.addWidget(self.input2)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_action)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def get_input_values(self):
        return self.input1.text(), self.input2.text() 

    def apply_action(self):
         self.accept()

class LinesPeak(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Determine No. Of Peaks")
        layout = QVBoxLayout()
        label1 = QLabel("Peaks Number:")
        self.input1 = QLineEdit()
    

        layout.addWidget(label1)
        layout.addWidget(self.input1)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_action)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def get_input_values(self):
        return self.input1.text()

    def apply_action(self):
         self.accept()

class CV_App(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up the UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  

        self.input_image = None
        self.input_image2 = None
        self.output_image = None
        self.hybrid_image1 = None
        self.hybrid_image2 = None
        self.gray_img = None
        self.gray_img2 = None
        self.filters_slider = None
        self.lpf_image = None
        self.hpf_image = None
        self.pf_hybrid_flag = True
        self.image_path = None

        self.ui.BrowseButton.clicked.connect(self.browse_img)
        self.ui.BrowseButton_2.clicked.connect(self.browse_input_image2)
        # self.ui.HighpassButton.clicked.connect(lambda: self.high_pass_filter(self.gray_img))
        # self.ui.LowpassButton.clicked.connect(lambda: self.low_pass_filter(self.gray_img))
        # self.ui.horizontalSlider.valueChanged.connect(self.set_cutoff_freq_value)
        self.ui.GenerateHybrid.clicked.connect(lambda: self.generate_hybrid_image(self.gray_img, self.gray_img2))
        self.ui.VerticalSlider.valueChanged.connect(self.set_cutoff_freq_value)

        self.ui.RefreshButton.clicked.connect(self.refresh_img)

        self.ui.AverageFilterButton.clicked.connect(self.average_filter)
        self.ui.GaussianFilterButton.clicked.connect(self.gaussian_filter)
        self.ui.MedianFilterButton.clicked.connect(self.median_filter)

        self.ui.UniformNoiseButton.clicked.connect(self.add_uniform_noise)
        self.ui.SaltPepperNoiseButton.clicked.connect(self.add_salt_and_pepper_noise)
        self.ui.GaussianNoiseButton.clicked.connect(self.add_gaussian_noise)

        self.ui.EqualizeButton.clicked.connect(self.image_equalization)
        self.ui.NormalizeButton.clicked.connect(self.image_normalization)
        self.ui.GlobalThresholdButton.clicked.connect(self.global_threshold)
        self.ui.LocalThresholdButton.clicked.connect(self.local_threshold)

        self.ui.CannyButton.clicked.connect(self.canny_edge_detection)
        self.ui.SobelButton.clicked.connect(self.perform_sobel_edge_detection)
        self.ui.RobetButton.clicked.connect(self.perform_roberts_edge_detection)
        self.ui.PrewittButton.clicked.connect(self.perform_prewitt_edge_detection)

        self.ui.LinesButton.clicked.connect(self.hough_lines)
        self.ui.CircleButton.clicked.connect(self.hough_circles)
        self.ui.ElipsesButton.clicked.connect(self.hough_elipses)

    def hough_lines(self):
        dialog = LinesPeak(self)
        if dialog.exec_():
            peaks_no = int(dialog.get_input_values())
        image = Image.open(self.image_path)
        final_img = detect_hough_lines(image, peaks=peaks_no)
        qImg = QImage(final_img.data, final_img.shape[1], final_img.shape[0], final_img.strides[0],QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.hough_output.setPixmap(pixmap)

    def hough_circles(self):
       # Detect circles and get the result image
        print("the type is:",type(self.input_image))
        print("the path is:",self.image_path)
        image_circles = hough_circles(self.image_path)
        print("the Circles is is is:",image_circles)
    
        # Convert the result image to RGB format
        image_circles = np.array(image_circles)  # Convert PIL.Image to numpy array
        image_rgb = cv2.cvtColor(image_circles, cv2.COLOR_BGR2RGB)

        # Create a QImage from the RGB image data
        qImg = QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], image_rgb.strides[0], QImage.Format_RGB888)

        # Convert QImage to QPixmap
        pixmap = QPixmap.fromImage(qImg)

        # Set the QPixmap to the label for display
        self.ui.hough_output.setPixmap(pixmap)

    def hough_elipses(self):
        image_elipse = draw_hough_elipses(self.image_path)
        qImg = QImage(image_elipse.data, image_elipse.shape[1], image_elipse.shape[0], image_elipse.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.hough_output.setPixmap(pixmap)

    def canny_edge_detection(self):
        if self.input_image is not None:
            canny_detector = CannyEdgeDetector(image=self.input_image,theta_resolution=1, rho_resolution=1, threshold=100 )
            canny_detector.detect_edges()
            edges = canny_detector.edges
            qImg = QImage(edges.data, edges.shape[1], edges.shape[0], edges.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)
            self.ui.EdgeDetection_outputImage.setPixmap(pixmap)
            
    def refresh_img(self):
        pixmap = QPixmap("grayscale_image.jpg" )
        self.ui.filter_outputImage.setPixmap(pixmap)
        self.ui.Threshold_outputImage.setPixmap(pixmap)
        self.ui.EdgeDetection_outputImage.setPixmap(pixmap)

    def add_uniform_noise(self, intensity=0.1):
        dialog = UniformNoiseDialog(self)
        if dialog.exec_():
            intensity = float(dialog.get_input_values())

        width, height = self.input_image.size
        image_gray = self.input_image.convert('L')
        noisy_image = np.array(image_gray)
        noise = np.random.uniform(-intensity, intensity, (height, width))
        noisy_image = np.clip(noisy_image + noise * 255, 0, 255).astype(np.uint8)

        output_image = Image.fromarray(noisy_image)
        self.output_image = output_image
        qt_image = QImage(output_image.tobytes(), output_image.size[0], output_image.size[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def add_gaussian_noise(self, mean=0, std=25):
        dialog = GaussianNoiseDialog(self)
        if dialog.exec_():
            mean, std = map(float, dialog.get_input_values())

        width, height = self.input_image.size
        image_gray = self.input_image.convert('L')
        noisy_image = np.array(image_gray)
        noise = np.random.normal(mean, std, (height, width))
        noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)

        output_image = Image.fromarray(noisy_image)
        self.output_image = output_image
        qt_image = QImage(output_image.tobytes(), output_image.size[0], output_image.size[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def add_salt_and_pepper_noise(self, ratio=0.01):
        dialog = SaltPepperDialog(self)
        if dialog.exec_():
            ratio = float(dialog.get_input_values())

        width, height = self.input_image.size
        image_gray = self.input_image.convert('L')
        noisy_image = np.array(image_gray)
        
        salt_pepper = np.random.rand(height, width)
        noisy_image[salt_pepper < ratio / 2] = 0
        noisy_image[salt_pepper > 1 - ratio / 2] = 255

        output_image = Image.fromarray(noisy_image)
        self.output_image = output_image
        qt_image = QImage(output_image.tobytes(), output_image.size[0], output_image.size[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def average_filter(self):
        if self.ui.Radio3x3Kernal.isChecked():
            kernel_size = 3
        else:
            kernel_size = 5

        if self.output_image:
            input_img = self.output_image
        else:
            input_img = self.input_image

        image_gray = input_img.convert('L')
        img_array = np.array(image_gray)
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        filtered_image_array = convolve2d(img_array, kernel, mode='same').astype(np.uint8)

        filtered_image = Image.fromarray(filtered_image_array) 
        self.output_image = filtered_image
        qt_image = QImage(filtered_image.tobytes(), filtered_image.size[0], filtered_image.size[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def median_filter(self):
        if self.ui.Radio3x3Kernal.isChecked():
            kernel_size = 3
        else:
            kernel_size = 5
        if self.output_image:
            input_img = self.output_image
        else:
            input_img = self.input_image

        image_gray = input_img.convert('L')
        img_array = np.array(image_gray)
        height, width = img_array.shape
        filtered_image = np.zeros_like(img_array)

        pad_size = kernel_size // 2
        padded_image = np.pad(img_array, pad_size, mode='constant')

        for i in range(pad_size, height + pad_size):
            for j in range(pad_size, width + pad_size):
                window = padded_image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
                filtered_image[i - pad_size, j - pad_size] = np.median(window)

        filtered_image = Image.fromarray(filtered_image.astype('uint8'))
        self.output_image = filtered_image
        qt_image = QImage(filtered_image.tobytes(), filtered_image.size[0], filtered_image.size[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.filter_outputImage.setPixmap(pixmap)

    def gaussian_filter(self, kernel_size=3, sigma=100):
        if self.ui.Radio3x3Kernal.isChecked():
            kernel_size = 3
        else:
            kernel_size = 5

        if self.output_image:
            input_img = self.output_image
        else:
            input_img = self.input_image

        image_gray = input_img.convert('L')
        img_array = np.array(image_gray)

        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2)/(2*sigma**2)), (kernel_size, kernel_size))
        kernel = kernel / np.sum(kernel)

        filtered_image = convolve(img_array, kernel)
        filtered_image = Image.fromarray(filtered_image.astype('uint8'))
        self.output_image = filtered_image
        qt_image = QImage(filtered_image.tobytes(), filtered_image.size[0], filtered_image.size[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.filter_outputImage.setPixmap(pixmap)
        
    def browse_img(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        self.input_image = Image.open(f"{filename}")
        self.image_path = filename

        if filename:
            pixmap = QPixmap(filename)
            if not pixmap.isNull():
                self.ui.filter_inputImage.setPixmap(pixmap)
                self.ui.Threshold_inputImage.setPixmap(pixmap)
                self.ui.EdgeDetection_inputImage.setPixmap(pixmap)
                self.ui.hough_input.setPixmap(pixmap)
                self.ui.hybridInputImage1.setPixmap(pixmap)
                self.ui.hough_output.setPixmap(pixmap)
                
                self.input_image_cv = cv2.imread(filename)
                self.gray_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

                cv2.imwrite("grayscale_image.jpg" , self.gray_img)
                pixmap = QPixmap("grayscale_image.jpg" )
                self.ui.filter_outputImage.setPixmap(pixmap)
                self.ui.Threshold_outputImage.setPixmap(pixmap)
                self.ui.EdgeDetection_outputImage.setPixmap(pixmap)
                self.ui.freqOutputImage1.setPixmap(pixmap)

        self.draw_rgb_histogram(self.input_image_cv)
        self.draw_rgb_disturb_curve(self.input_image_cv)

    def browse_input_image2(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        self.input_image2 = Image.open(f"{filename}")

        if filename:
            pixmap = QPixmap(filename)
            if not pixmap.isNull():
                self.ui.hybridInputImage2.setPixmap(pixmap)
                
                self.input_image2_cv = cv2.imread(filename)
                self.gray_img2 = cv2.cvtColor(self.input_image2_cv, cv2.COLOR_BGR2GRAY)

                cv2.imwrite("grayscale_image.jpg" , self.gray_img2)
                pixmap = QPixmap("grayscale_image.jpg" )
                self.ui.freqOutputImage2.setPixmap(pixmap)

    def image_equalization(self):
        histogram, bins = np.histogram(self.gray_img.flatten(), 256, [0,256])

        cdf1 = histogram.cumsum()

        cdf_m = np.ma.masked_equal(cdf1, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        eq_img = cdf[self.gray_img]
        equalized_img = cv2.cvtColor(eq_img, cv2.COLOR_BGR2RGB)
        height, width, channel = equalized_img.shape
        bytesPerLine = 3 * width

        qImg = QImage(equalized_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.Threshold_outputImage.clear
        self.ui.Threshold_outputImage.setPixmap(pixmap)

    def image_normalization(self):
        dialog = ThresDialog(self)
        if dialog.exec_():
            threshold = float(dialog.get_input_values())

        lmin = float(self.gray_img.min())
        lmax = float(self.gray_img.max())
        x = (self.gray_img - lmin)
        y = (lmax - lmin)
        normalized_img = ((x / y) * threshold)
        cv2.imwrite("normalized_img.jpg" , normalized_img)
        
        height, width = self.gray_img.shape  
        qImg = QImage(normalized_img.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.Threshold_outputImage.setPixmap(pixmap)
        
    def global_threshold(self, threshold=200):
        dialog = ThresDialog(self)
        if dialog.exec_():
            threshold = float(dialog.get_input_values())

        thresholded_image = np.zeros_like(self.gray_img)
        thresholded_image[self.gray_img > threshold] = 220
        height, width = self.gray_img.shape

        qImg = QImage(thresholded_image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.Threshold_outputImage.setPixmap(pixmap)

    def local_threshold(self, block_size=3, threshold=80):
        if self.ui.Radio3x3Kernal_2.isChecked():
            block_size = 3
        else:
            block_size = 5

        dialog = ThresDialog(self)
        if dialog.exec_():
            threshold = float(dialog.get_input_values())

        thresholded_image = np.zeros_like(self.gray_img)
        padded_image = np.pad(self.gray_img, block_size//2, mode='constant')
        height, width = self.gray_img.shape
        for i in range(self.gray_img.shape[0]):
            for j in range(self.gray_img.shape[1]):
                neighborhood = padded_image[i:i+block_size, j:j+block_size]
                mean_value = np.mean(neighborhood)
                thresholded_image[i, j] = 255 if (self.gray_img[i, j] - mean_value) > threshold else 0
        
        qImg = QImage(thresholded_image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.Threshold_outputImage.setPixmap(pixmap)
    
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
         
    def sobel_edge_detection(self, image):
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

            # Normalize gradient magnitude to [0, 255]
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
            return gradient_magnitude.astype(np.uint8)  
    def active_contour(self, img, alpha, beta, w_line, w_edge):
        def convolve(image, kernel):
            if image is not None:
                height, width = image.shape
                k_height, k_width = kernel.shape
                output = np.zeros_like(image)
                padded_image = np.pad(image, ((k_height//2, k_height//2), (k_width//2, k_width//2)), mode='constant')

                for i in range(height):
                    for j in range(width):
                        output[i, j] = np.sum(padded_image[i:i+k_height, j:j+k_width] * kernel)
                return output

        def sobel_edge_detection(image):
            if image is not None:
            
                image = image.astype(np.float32)

                # Sobel filter kernels
                kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

                # Convolve image with Sobel kernels to calculate gradients
                grad_x = convolve(image, kernel_x)
                grad_y = convolve(image, kernel_y)

                # Compute gradient magnitude
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

                # Normalize gradient magnitude to [0, 255]
                gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
                edge = gradient_magnitude.astype(np.uint8) 
                return edge 

        def calculate_derivatives(snake):
            N = snake.shape[0]
            first_derivatives = np.zeros((N, 2))
            second_derivatives = np.zeros((N, 2))

            # Calculate first derivatives using central differences
            first_derivatives[1:-1] = (snake[2:] - snake[:-2]) / 2.0
            first_derivatives[0] = snake[1] - snake[0]
            first_derivatives[-1] = snake[-1] - snake[-2]

            # Calculate second derivatives using central differences
            second_derivatives[1:-1] = snake[:-2] - 2 * snake[1:-1] + snake[2:]
            second_derivatives[0] = snake[0] - 2 * snake[1] + snake[2]
            second_derivatives[-1] = snake[-3] - 2 * snake[-2] + snake[-1]

            return first_derivatives, second_derivatives

        def gaussian_filter(self, img,  kernel_size=3, sigma=100):
            kernel_size = 3
            img_array = np.array(img)
            kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2)/(2*sigma**2)), (kernel_size, kernel_size))
            kernel = kernel / np.sum(kernel)
            filtered_image = convolve(img_array, kernel)
            filtered_image = Image.fromarray(filtered_image.astype('uint8'))
            return filtered_image

        img = gaussian_filter(img)
        
        s = np.linspace(0, 2*np.pi, 400)
        r = 100 + 100*np.sin(s)
        c = 220 + 100*np.cos(s)
        snake = np.array([r, c]).T
        D_1, D_2 = calculate_derivatives(snake)

        E_internal = 0.5 * ((alpha * (D_1)**2 ) + (beta * (D_2)**2 ))
        E_img = (w_line*img) + (w_edge*sobel_edge_detection(img))
        E_snake = E_internal + E_img

    def perform_sobel_edge_detection(self):
        if self.gray_img is not None:
            gradient_magnitude = self.sobel_edge_detection(self.gray_img)
            if gradient_magnitude is not None:
                qImg = QImage(gradient_magnitude.data, gradient_magnitude.shape[1], gradient_magnitude.shape[0], QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qImg)
                self.ui.EdgeDetection_outputImage.setPixmap(pixmap)

    def roberts_edge_detection(self, image):
        if image is not None:
            # Convert the image to floating point
            image = image.astype(np.float32)

            # Roberts Cross kernels
            kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
            
            # Convolve image with Roberts kernels to calculate gradients
            grad_x = self.convolve(image, kernel_x)
            grad_y = self.convolve(image, kernel_y)

            # Compute gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Normalize gradient magnitude to [0, 255]
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

            return gradient_magnitude.astype(np.uint8)  # Convert back to uint8 for image display

    def perform_roberts_edge_detection(self):
        if self.gray_img is not None:
            gradient_magnitude = self.roberts_edge_detection(self.gray_img)
            if gradient_magnitude is not None:
                qImg = QImage(gradient_magnitude.data, gradient_magnitude.shape[1], gradient_magnitude.shape[0], QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qImg)
                self.ui.EdgeDetection_outputImage.setPixmap(pixmap)

    def prewitt_edge_detection(self, image):
        if image is not None:
            # Convert the image to floating point
            image = image.astype(np.float32)

            # Prewitt kernels
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

            # Convolve image with Prewitt kernels to calculate gradients
            grad_x = self.convolve(image, kernel_x)
            grad_y = self.convolve(image, kernel_y)

            # Compute gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Normalize gradient magnitude to [0, 255]
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

            return gradient_magnitude.astype(np.uint8)  # Convert back to uint8 for image display

    def perform_prewitt_edge_detection(self):
        if self.gray_img is not None:
            gradient_magnitude = self.prewitt_edge_detection(self.gray_img)
            if gradient_magnitude is not None:
                qImg = QImage(gradient_magnitude.data, gradient_magnitude.shape[1], gradient_magnitude.shape[0], QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qImg)
                self.ui.EdgeDetection_outputImage.setPixmap(pixmap)
    
    def draw_rgb_histogram(self,image):
        red_channel = image[:,:,0]
        green_channel = image[:,:,1]
        blue_channel = image[:,:,2]

        red_hist, red_bins = np.histogram(red_channel.flatten(), bins=256, range=[0, 256])
        green_hist, green_bins = np.histogram(green_channel.flatten(), bins=256, range=[0, 256])
        blue_hist, blue_bins = np.histogram(blue_channel.flatten(), bins=256, range=[0, 256])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(red_bins[:-1], red_hist, color='red', label='Red')
        ax.set_title("Red Histogram")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        fig.savefig("assets/graphs/red_histogram.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(green_bins[:-1], green_hist, color='green', label='Green')
        ax.set_title("Green Histogram")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        fig.savefig("assets/graphs/green_histogram.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(blue_bins[:-1], blue_hist, color='blue', label='Blue')
        ax.set_title("Blue Histogram")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        fig.savefig("assets/graphs/blue_histogram.png")

        # Convert saved images to QPixmaps and set them to QLabels
        red_pixmap = QPixmap("assets/graphs/red_histogram.png")
        green_pixmap = QPixmap("assets/graphs/green_histogram.png")
        blue_pixmap = QPixmap("assets/graphs/blue_histogram.png")

        # Set the QPixmap to the corresponding QLabel
        self.ui.Histogram_1.setPixmap(red_pixmap)
        self.ui.Histogram_3.setPixmap(green_pixmap)
        self.ui.Histogram_5.setPixmap(blue_pixmap)

    def draw_rgb_disturb_curve(self,image):
        red_channel = image[:,:,0]
        green_channel = image[:,:,1]
        blue_channel = image[:,:,2]

        # Calculate histograms for each channel
        red_hist, red_bins = np.histogram(red_channel.flatten(), bins=256, range=[0, 256])
        green_hist, green_bins = np.histogram(green_channel.flatten(), bins=256, range=[0, 256])
        blue_hist, blue_bins = np.histogram(blue_channel.flatten(), bins=256, range=[0, 256])

        red_cum_hist = np.cumsum(red_hist)
        green_cum_hist = np.cumsum(green_hist)
        blue_cum_hist = np.cumsum(blue_hist)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(red_bins[:-1], red_cum_hist, color='red', label='Red')
        ax.set_title("Red CDF")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        fig.savefig("assets/graphs/red_cdf.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(green_bins[:-1], green_cum_hist, color='green', label='Green')
        ax.set_title("Green CDF")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        fig.savefig("assets/graphs/green_cdf.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(blue_bins[:-1], blue_cum_hist, color='blue', label='Blue')
        ax.set_title("Blue CDF")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        fig.savefig("assets/graphs/blue_cdf.png")

        red_pixmap = QPixmap("assets/graphs/red_cdf.png")
        green_pixmap = QPixmap("assets/graphs/green_cdf.png")
        blue_pixmap = QPixmap("assets/graphs/blue_cdf.png")

        # Set the QPixmap to the corresponding QLabel
        self.ui.Histogram_2.setPixmap(red_pixmap)
        self.ui.Histogram_4.setPixmap(green_pixmap)
        self.ui.Histogram_6.setPixmap(blue_pixmap)

    # def draw_distribution_curve(self,image):
    #     # Calculate histogram
    #     histogram, bins = np.histogram(image.flatten(), bins=256, range=[0,256])

    #     # Cumulative distribution function (CDF)
    #     cdf = histogram.cumsum()
    #     cdf_normalized = cdf * histogram.max() / cdf.max()

    #     # Plot distribution curve
    #     plt.figure(figsize=(8, 6))
    #     plt.bar(bins[:-1], cdf_normalized, color='b',align="edge")
    #     plt.title("Distribution Curve")
    #     plt.xlabel("Pixel Intensity")
    #     plt.ylabel("CDF")

    #     plt.tight_layout()
    #     plt.savefig("assets/graphs/disturb_curve.png")
    #     pixmap = QPixmap("assets/graphs/disturb_curve.png")
    #     self.ui.Histogram_2.setPixmap(pixmap)
            
    # def draw_histogram(self,image):
    #     # Calculate histogram
    #     histogram, bins = np.histogram(image.flatten(), bins=256, range=[0,256])

    #     # Plot histogram
    #     plt.figure(figsize=(8, 6))
    #     plt.bar(bins[:-1], histogram, width=1)
    #     plt.title("Histogram")
    #     plt.xlabel("Pixel Intensity")
    #     plt.ylabel("Frequency")

    #     plt.tight_layout()
    #     plt.savefig("assets/graphs/histogram.png")
    #     pixmap = QPixmap("assets/graphs/histogram.png")

    #     self.ui.Histogram_1.setPixmap(pixmap)
    

    def low_pass_filter(self, image):
        if image is None:
            return

        image_array = np.array(image)

        # Compute the FFT of the image
        fft_image = np.fft.fft2(image_array)
        
        # Handling the value of cutoff freq based on the mode (pass filter or hybrid)
        if self.pf_hybrid_flag:
            self.filters_slider = self.ui.horizontalSlider.value()
        else:
            self.filters_slider = self.ui.VerticalSlider.value()
        
        # Create a filter mask based on the cut-off frequency
        cutoff_freq = self.filters_slider
        rows, cols = image_array.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols), np.uint8)
        mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 0

        # Apply the filter mask to the frequency domain representation of the image
        fft_image_lpf = fft_image * mask

        # Compute the inverse FFT to obtain the filtered image in the spatial domain
        filtered_image = np.fft.ifft2(fft_image_lpf).real.astype(np.uint8)
        self.lpf_image = Image.fromarray(filtered_image)
        
        # Handling what data to be returned (if in pass filter mode we show image, but in hybrid mode we return an fft array)
        if self.pf_hybrid_flag:
            qt_image = QImage(self.lpf_image.tobytes(), self.lpf_image.size[0], self.lpf_image.size[1], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qt_image)
            self.ui.filter_outputImage.setPixmap(pixmap)
            self.ui.pass_outputImage.setPixmap(pixmap)
        else:
            return fft_image_lpf

    # Highpass Function 
    def high_pass_filter(self, image):
        if image is None:
            return

        image_array = np.array(image)
        # Compute the FFT of the image
        fft_image = np.fft.fft2(image_array)

        # Handling the value of cutoff freq based on the mode (pass filter or hybrid)
        if self.pf_hybrid_flag:
            self.filters_slider = self.ui.horizontalSlider.value()
        else:
            self.filters_slider = self.ui.VerticalSlider.value()

        # Create a filter mask based on the cut-off frequency for high pass filtering
        cutoff_freq = self.filters_slider
        rows, cols = image_array.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 1

        # Apply the filter mask to the frequency domain representation of the image
        fft_image_hpf = fft_image * mask

        # Compute the inverse FFT to obtain the filtered image in the spatial domain
        filtered_image2 = np.fft.ifft2(fft_image_hpf).real.astype(np.uint8)
        self.hpf_image = Image.fromarray(filtered_image2)
        
        # Handling what data to be returned (if in pass filter mode we show image, but in hybrid mode we return an fft array)
        if self.pf_hybrid_flag:
            qt_image = QImage(self.hpf_image.tobytes(), self.hpf_image.size[0], self.hpf_image.size[1], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qt_image)
            self.ui.filter_outputImage.setPixmap(pixmap)
            self.ui.pass_outputImage.setPixmap(pixmap)
        else:
            return fft_image_hpf

    # Hybrid Function 
    def generate_hybrid_image(self, input1, input2):

        if input1 is None or input2 is None:
            QMessageBox.warning(self, "Warning", "Please load two images first.")
            return

        # Handling the flag to use cutoff freq of hybrid mode silder
        self.pf_hybrid_flag = False

        # Compute low pass filter for the input images
        lowpass_image1 = self.low_pass_filter(input1)
        lowpass_image2 = self.low_pass_filter(input2)

        # Compute high pass filter for the input images
        highpass_image1 = self.high_pass_filter(input1)
        highpass_image2 = self.high_pass_filter(input2)


        # Resize highpassed images to match the dimensions of lowpassed images
        highpass_image2_real = np.real(highpass_image2)
        highpass_image2_imag = np.imag(highpass_image2)
        highpass_image2_real_resized = cv2.resize(highpass_image2_real, (lowpass_image1.shape[1], lowpass_image1.shape[0]))
        highpass_image2_imag_resized = cv2.resize(highpass_image2_imag, (lowpass_image1.shape[1], lowpass_image1.shape[0]))
        highpass_image2_resized = highpass_image2_real_resized + 1j * highpass_image2_imag_resized

        highpass_image1_real = np.real(highpass_image1)
        highpass_image1_imag = np.imag(highpass_image1)
        highpass_image1_real_resized = cv2.resize(highpass_image1_real, (lowpass_image2.shape[1], lowpass_image2.shape[0]))
        highpass_image1_imag_resized = cv2.resize(highpass_image1_imag, (lowpass_image2.shape[1], lowpass_image2.shape[0]))
        highpass_image1_resized = highpass_image1_real_resized + 1j * highpass_image1_imag_resized

        # Creating 2 arrays of the hybrid images, 1st is of lowpass of the 1st input and highpass of the 2nd input and the other array is vice versa
        hybrid_image1_fft_array = lowpass_image1 + highpass_image2_resized
        hybrid_image2_fft_array = lowpass_image2 + highpass_image1_resized

        # Converting arrays from frequency domain to spatial domain then to images
        inverse_hybrid1 = np.fft.ifft2(hybrid_image1_fft_array).real.astype(np.uint8)
        inverse_hybrid2 = np.fft.ifft2(hybrid_image2_fft_array).real.astype(np.uint8)
        self.hybrid_image1 = Image.fromarray(inverse_hybrid1)
        self.hybrid_image2 = Image.fromarray(inverse_hybrid2)

        # Handling which hybrid image to show
        if self.ui.radioButton_2.isChecked() :
            qt_image = QImage(self.hybrid_image1.tobytes(), self.hybrid_image1.size[0], self.hybrid_image1.size[1], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qt_image)
            self.ui.finalHybridImage.setPixmap(pixmap)

        else:
            qt_image = QImage(self.hybrid_image2.tobytes(), self.hybrid_image2.size[0], self.hybrid_image2.size[1], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qt_image)
            self.ui.finalHybridImage.setPixmap(pixmap)
        
        # Handling the flag to indicate end of the hybrid process
        self.pf_hybrid_flag = True

    # Handling labels of sliders Function 
    def set_cutoff_freq_value(self):
        self.cutoff_freq_value = int(self.ui.horizontalSlider.value())
        self.ui.label_5.setText(f"{self.cutoff_freq_value}")
        self.cutoff_freq_value_hybrid = int(self.ui.VerticalSlider.value())
        self.ui.label_7.setText(f"{self.cutoff_freq_value_hybrid}")
    



    def hough_ellipse(self):
    
        # Canny Edge Detection using OpenCV
        edges = cv2.Canny(self.gray_img, 50, 150)

        # Find edge points
        edge_points = np.transpose(np.nonzero(edges))

        # Get the shape of the edges image
        edges_shape = np.shape(edges)

        # Initialize accumulator array
        accumulator = np.zeros(edges_shape, dtype=np.uint)

        # Define parameter space
        a_max = max(edges_shape) // 2
        a_range = np.arange(-a_max, a_max + 1)
        b_max = min(edges_shape) // 2
        b_range = np.arange(-b_max, b_max + 1)

        # Iterate through edge points and vote for potential ellipses
        for x, y in edge_points:
            for a in a_range:
                for b in b_range:
                    if a != 0 and b != 0:  # Check if a and b are non-zero
                        new_x = x + a
                        if new_x < 0 or new_x >= edges_shape[0]:
                            continue

                        new_y = y + b
                        if new_y < 0 or new_y >= edges_shape[1]:
                            continue

                        accumulator[new_x, new_y] += 1

        # Threshold accumulator to find potential ellipses
        threshold = 100  
        ellipses = np.argwhere(accumulator > threshold)

        # Convert to ellipse parameters (center and radii)
        detected_ellipses = []
        for x, y in ellipses:
            detected_ellipses.append((x, y, a_range[y], b_range[x]))

        # Superimpose detected ellipses on the original image
        result_image = QPixmap(self.input_image)
        painter = QPainter(result_image)
        painter.setPen(QColor(0, 255, 0))

        for ellipse in detected_ellipses:
            painter.drawEllipse(*ellipse)
        painter.end()

        self.ui.hough_output.setPixmap(result_image)

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CV_App()
    window.setWindowTitle("Task 1")
    window.resize(1450,950)
    window.show() 
    sys.exit(app.exec_())
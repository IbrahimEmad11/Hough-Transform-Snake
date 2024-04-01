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
from active_contour import ActiveContour
from snake_contour import *


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
        self.output_image = None
        self.gray_img = None
        self.image_path = None
        self.alpha=0
        self.beta=0
        self.gamma=10
        self.iterations=1
        self.sigma=1

        self.ui.BrowseButton.clicked.connect(self.browse_img)

        self.ui.RefreshButton.clicked.connect(self.refresh_img)

        self.ui.SobelButton.clicked.connect(self.perform_sobel_edge_detection)
        self.ui.RobetButton.clicked.connect(self.perform_roberts_edge_detection)
        self.ui.PrewittButton.clicked.connect(self.perform_prewitt_edge_detection)

        self.ui.LinesButton.clicked.connect(self.hough_lines)
        self.ui.CircleButton.clicked.connect(self.hough_circles)
        self.ui.ElipsesButton.clicked.connect(self.hough_elipses)

        self.ui.ActiveContourButton.clicked.connect(self.draw_active)

        self.ui.alphaSlider.valueChanged.connect(self.alpha_changed)
        self.ui.betaSlider.valueChanged.connect(self.beta_changed)
        self.ui.gammaSlider.valueChanged.connect(self.gamma_changed)
        self.ui.iterationSlider.valueChanged.connect(self.iteration_changed)
        self.ui.sigmaSlider.valueChanged.connect(self.sigma_changed)

    def alpha_changed(self, value):
        self.alpha = value/100

    def beta_changed(self, value):
        self.beta = value/100

    def gamma_changed(self, value):
        self.gamma = value

    def iteration_changed(self, value):
        self.iterations = value  

    def sigma_changed(self, value):
        self.sigma = value

    def draw_active(self):
        og_image = Image.open(self.image_path)
        gray_image = ImageOps.grayscale(og_image)
        resized_gray_img = resize_img(gray_image)
        np_image = np.array(gray_image)
        resized_np_image = np.array(resized_gray_img)
        area,perimeter=activeContour(resized_np_image,self.alpha,self.beta,self.gamma,self.iterations,self.sigma)
        st.image("assets/contour/active_contour.jpg")
        st.write("Area = ",area)
        st.write("perimeter = ",perimeter)
        output_cont = QPixmap('assets/contour/active_contour.jpg')
        self.ui.activeContour_outputImage.setPixmap(output_cont)


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
        print("the Circles equal:",image_circles)
        # Convert the result image to RGB format
        image_circles = np.array(image_circles)  # Convert PIL.Image to numpy array
        image_rgb = cv2.cvtColor(image_circles, cv2.COLOR_BGR2RGB)
        qImg = QImage(image_rgb.data, image_rgb.shape[1], image_rgb.shape[0], image_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.hough_output.setPixmap(pixmap)

    def hough_elipses(self):
        image_elipse = draw_hough_elipses(self.image_path)
        qImg = QImage(image_elipse.data, image_elipse.shape[1], image_elipse.shape[0], image_elipse.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.hough_output.setPixmap(pixmap)

    def canny_edge_detection(self):
        if self.input_image is not None:
            canny_detector = CannyEdgeDetector(image=self.input_image)
            canny_detector.detect_edges()
            edges = canny_detector.edges
            qImg = QImage(edges.data, edges.shape[1], edges.shape[0], edges.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)
            self.ui.EdgeDetection_outputImage.setPixmap(pixmap)
            
    def browse_img(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        self.input_image = Image.open(f"{filename}")
        self.image_path = filename

        if filename:
            pixmap = QPixmap(filename)
            if not pixmap.isNull():
                self.ui.EdgeDetection_inputImage.setPixmap(pixmap)
                self.ui.hough_input.setPixmap(pixmap)
                self.ui.activeContour_inputImage.setPixmap(pixmap)
                self.ui.hough_output.setPixmap(pixmap)
                
                self.input_image_cv = cv2.imread(filename)
                self.gray_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

                cv2.imwrite("grayscale_image.jpg" , self.gray_img)
                pixmap = QPixmap("grayscale_image.jpg" )
                self.ui.EdgeDetection_outputImage.setPixmap(pixmap)

    def refresh_img(self):
        pixmap = QPixmap("grayscale_image.jpg" )
        self.ui.activeContour_outputImage.setPixmap(pixmap)
        self.ui.EdgeDetection_outputImage.setPixmap(pixmap)
        

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
    
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CV_App()
    window.setWindowTitle("Task 1")
    window.resize(1450,950)
    window.show() 
    sys.exit(app.exec_())
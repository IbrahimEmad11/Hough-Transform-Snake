import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QSlider , QColorDialog, QAction, QTextEdit, QMessageBox
from PyQt5.QtCore import QTimer,Qt, QPointF
from PyQt5.QtGui import QColor, QIcon, QCursor, QKeySequence, QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QProgressBar, QDialog, QVBoxLayout, QLineEdit, QLabel
from task1 import Ui_MainWindow
from PIL import Image , ImageFilter
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# pyuic5 task1.ui -o task1.py
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


class CV_App(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set up the UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  
        self.input_image = None
        self.output_image = None
        self.input_image2 = None
        self.hybrid_image1 = None
        self.hybrid_image2 = None
        self.gray_img = None
        self.ui.BrowseButton.clicked.connect(self.browse_img)
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

        if filename:
            pixmap = QPixmap(filename)
            if not pixmap.isNull():
                self.ui.filter_inputImage.setPixmap(pixmap)
                self.ui.Threshold_inputImage.setPixmap(pixmap)
                self.ui.EdgeDetection_inputImage.setPixmap(pixmap)
                
                self.input_image_cv = cv2.imread(filename)
                self.gray_img = cv2.cvtColor(self.input_image_cv, cv2.COLOR_BGR2GRAY)

                cv2.imwrite("grayscale_image.jpg" , self.gray_img)
                pixmap = QPixmap("grayscale_image.jpg" )
                self.ui.filter_outputImage.setPixmap(pixmap)
                self.ui.Threshold_outputImage.setPixmap(pixmap)
                self.ui.EdgeDetection_outputImage.setPixmap(pixmap)

        self.draw_histogram(self.gray_img)
        self.draw_distribution_curve(self.gray_img)

    # Equalization
    def image_equalization(self):
        # Compute histogram of the original image
        histogram, bins = np.histogram(self.gray_img.flatten(), 256, [0,256])

        # Compute cumulative distribution function (CDF)
        cdf1 = histogram.cumsum()

        # Apply histogram equalization using the CDF
        cdf_m = np.ma.masked_equal(cdf1, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        # Map original pixel intensities to new intensities using the CDF
        eq_img = cdf[self.gray_img]
        equalized_img = cv2.cvtColor(eq_img, cv2.COLOR_BGR2RGB)
        height, width, channel = equalized_img.shape
        bytesPerLine = 3 * width
        qImg = QImage(equalized_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.Threshold_outputImage.clear
        self.ui.Threshold_outputImage.setPixmap(pixmap)

    ##########################################################################################################################################

    # Normalization
    def image_normalization(self):
        lmin = float(self.gray_img.min())
        lmax = float(self.gray_img.max())
        x = (self.gray_img - lmin)
        y = (lmax - lmin)
        normalized_img = ((x / y) * 3)
        
        height, width = self.gray_img.shape  
        qImg = QImage(normalized_img.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        self.ui.Threshold_outputImage.clear()
        self.ui.Threshold_outputImage.setPixmap(pixmap)

    
        
    def global_threshold(self, threshold=200):

        thresholded_image = np.zeros_like(self.gray_img)
        thresholded_image[self.gray_img > threshold] = 220
        height, width = self.gray_img.shape
        qImg = QImage(thresholded_image.data, width, height, width, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(qImg)
        self.ui.Threshold_outputImage.clear()
        self.ui.Threshold_outputImage.setPixmap(pixmap)

    def local_threshold(self, block_size=3, constant=80):
        thresholded_image = np.zeros_like(self.gray_img)
        padded_image = np.pad(self.gray_img, block_size//2, mode='constant')
        height, width = self.gray_img.shape
        for i in range(self.gray_img.shape[0]):
            for j in range(self.gray_img.shape[1]):
                neighborhood = padded_image[i:i+block_size, j:j+block_size]
                mean_value = np.mean(neighborhood)
                thresholded_image[i, j] = 255 if (self.gray_img[i, j] - mean_value) > constant else 0
        
        qImg = QImage(thresholded_image.data, width, height, width, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(qImg)
        self.ui.Threshold_outputImage.clear()
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
        
    def canny_edge_detection(self):
        if self.input_image is not None:
            edges = cv2.Canny(self.gray_img, 100, 200)
            qImg = QImage(edges.data, edges.shape[1], edges.shape[0], edges.strides[0], QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)
            self.ui.EdgeDetection_outputImage.setPixmap(pixmap)
    
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
            kernel_x = np.array([[1, 0], [0, -1]])
            kernel_y = np.array([[0, 1], [-1, 0]])

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
    
    def draw_histogram(self,image):
        # Calculate histogram
        histogram, bins = np.histogram(image.flatten(), bins=256, range=[0,256])

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.bar(bins[:-1], histogram, width=1)
        plt.title("Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig("assets/graphs/histogram.png")
        pixmap = QPixmap("assets/graphs/histogram.png")

        self.ui.Histogram_1.setPixmap(pixmap)
        
    
    def draw_distribution_curve(self,image):
        # Calculate histogram
        histogram, bins = np.histogram(image.flatten(), bins=256, range=[0,256])

        # Cumulative distribution function (CDF)
        cdf = histogram.cumsum()
        cdf_normalized = cdf * histogram.max() / cdf.max()

        # Plot distribution curve
        plt.figure(figsize=(8, 6))
        plt.bar(bins[:-1], cdf_normalized, color='b',align="edge")
        plt.title("Distribution Curve")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("CDF")

        plt.tight_layout()
        plt.savefig("assets/graphs/disturb_curve.png")
        pixmap = QPixmap("assets/graphs/disturb_curve.png")
        self.ui.Histogram_2.setPixmap(pixmap)
            
    

############################################################### Requirement no. 9 ###############################################################

    def high_pass_filter(self, image):
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])

        hpf_image = self.apply_filter(image, kernel)
        return hpf_image

    def low_pass_filter(self, image):
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]) / 9

        lpf_image = self.apply_filter(image, kernel)
        return lpf_image

    def apply_filter(self, image, kernel):
        input_array = np.array(image.convertToFormat(QImage.Format_Grayscale8))
        processed_array = convolve(input_array, kernel)
        processed_array = np.clip(processed_array, 0, 255).astype(np.uint8)
        filtered_image = QImage(processed_array.data, processed_array.shape[1], processed_array.shape[0],
                                 processed_array.shape[1], QImage.Format_Grayscale8)
        return filtered_image
    


############################################################### Requirement no. 10 ###############################################################

    def generate_hybrid_images(self):
        if self.input_image is None or self.input_image2 is None:
            QMessageBox.warning(self, "Warning", "Please load two images first.")
            return

        # Apply high-pass and low-pass filters to both images
        high_pass_image1 = self.apply_high_pass_filter(self.input_image)
        high_pass_image2 = self.apply_high_pass_filter(self.input_image2)
        low_pass_image1 = self.apply_low_pass_filter(self.input_image)
        low_pass_image2 = self.apply_low_pass_filter(self.input_image2)

        #Create 1st hybrid image:
        hybrid_array1 = np.array(low_pass_image1) + np.array(high_pass_image2)
        hybrid_array1 = np.clip(hybrid_array1, 0, 255).astype(np.uint8)

        self.hybrid_image1 = QImage(hybrid_array1.data, hybrid_array1.shape[1], hybrid_array1.shape[0],
                              hybrid_array1.shape[1], QImage.Format_Grayscale8)
        #Create 2nd hybrid image:
        hybrid_array2 = np.array(low_pass_image2) + np.array(high_pass_image1)
        hybrid_array2 = np.clip(hybrid_array2, 0, 255).astype(np.uint8)

        self.hybrid_image2 = QImage(hybrid_array2.data, hybrid_array2.shape[1], hybrid_array2.shape[0],
                              hybrid_array2.shape[1], QImage.Format_Grayscale8)

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CV_App()
    window.setWindowTitle("Task 1")
    window.resize(1450,950)
    window.show() 
    sys.exit(app.exec_())
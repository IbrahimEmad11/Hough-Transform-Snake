import numpy as np
import matplotlib.pyplot as plt
from canny_detection import CannyEdgeDetector
import cv2
from PIL import Image
from collections import defaultdict

def hough_peaks(H, peaks, neighborhood_size=3):
  
    indices = []
    H1 = np.copy(H)
    
    # loop through number of peaks to identify
    for i in range(peaks):
        idx = np.argmax(H1)  # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape)  # remap to shape of H to be 2d array
        indices.append(H1_idx)

        idx_y, idx_x = H1_idx  # separate x, y indices from argmax(H)
        
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (neighborhood_size / 2)) < 0:
            min_x = 0
        else:
            min_x = idx_x - (neighborhood_size / 2)
        if (idx_x + (neighborhood_size / 2) + 1) > H.shape[1]:
            max_x = H.shape[1]
        else:
            max_x = idx_x + (neighborhood_size / 2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (neighborhood_size / 2)) < 0:
            min_y = 0
        else:
            min_y = idx_y - (neighborhood_size / 2)
        if (idx_y + (neighborhood_size / 2) + 1) > H.shape[0]:
            max_y = H.shape[0]
        else:
            max_y = idx_y + (neighborhood_size / 2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if x == min_x or x == (max_x - 1):
                    H[y, x] = 255
                if y == min_y or y == (max_y - 1):
                    H[y, x] = 255

    # return the indices and the original Hough space with selected points
    return indices, H

def hough_lines_draw(img, indices, rhos, thetas):

    for i in range(len(indices)):
        # get lines from rhos and thetas
        rho = rhos[indices[i][0]]
        theta = thetas[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # these are then scaled so that the lines go off the edges of the image
        y1 = int(y0 + 1000 * (a))
        x1 = int(x0 + 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

def line_detection(image: np.ndarray,T_low,T_upper):

    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg, (5,5), 1.5)
    edgeImg = cv2.Canny(blurImg, T_low, T_upper)
    

    height, width = edgeImg.shape
    
    maxDist = int(np.around(np.sqrt(height**2 + width**2)))
    
    thetas = np.deg2rad(np.arange(-90, 90))
    rhos = np.linspace(-maxDist, maxDist, 2*maxDist)
    
    accumulator = np.zeros((2 * maxDist, len(thetas)))
    
    for y in range(height):
        for x in range(width):
            if edgeImg[y,x] > 0:
                for k in range(len(thetas)):
                    r = x * np.cos(thetas[k]) + y * np.sin(thetas[k])
                    accumulator[int(r) + maxDist, k] += 1
                    
    return accumulator, thetas, rhos

def detect_hough_lines(source: np.ndarray,T_low=20,T_high=100,peaks: int = 10):
    
    src = np.copy(source)
    H, thetas, rhos = line_detection(src,T_low,T_high)
    indicies, H = hough_peaks(H, peaks) 
    hough_lines_draw(src, indicies, rhos, thetas)
    plt.imshow(src)
    plt.axis("off")
    return src
###################################################################### HOUGH CIRCLES ####################################################################


def calculateAccumlator(img_height, img_width, edge_image, circle_candidates):
    accumulator = defaultdict(int)
    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0: #white pixel
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    accumulator[(x_center, y_center, r)] += 1 #vote for current candidate
    return accumulator

def condidate_circles(thetas, rs, num_thetas):

    # Calculate Cos(theta) and Sin(theta) it will be required later
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    circle_candidates = []

    for r in rs:
        for t in range(num_thetas):
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
    return circle_candidates

def post_procces(out_circles, pixel_threshold):
    postprocess_circles = []
    for x, y, r, v in out_circles:

      # Remove nearby duplicate circles based on pixel_threshold
      if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_circles):
        postprocess_circles.append((x, y, r, v))
    return postprocess_circles

def hough_circle(circle_color,img_path:str, r_min:int = 20, r_max:int = 100, delta_r:int = 1, num_thetas:int = 100, bin_threshold:float = 0.4, min_edge_threshold:int = 100, max_edge_threshold:int = 200, pixel_threshold:int = 20,  post_process:bool = True):
        
    input_img = cv2.imread(img_path)

    #Edge detection on the input image
    edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)

    if edge_image is None:
        print ("Error in input image!")
        return

    #image size
    img_height, img_width = edge_image.shape[:2]
    
    # R and Theta ranges
    dtheta = int(360 / num_thetas)
    
    ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
    thetas = np.arange(0, 360, step=dtheta)
    
    ## Radius ranges from r_min to r_max 
    rs = np.arange(r_min, r_max, step=delta_r)
    
    circle_candidates = condidate_circles(thetas, rs, num_thetas)

    accumulator = calculateAccumlator(img_height, img_width, edge_image, circle_candidates)
    
    # Output image with detected lines drawn
    output_img = input_img.copy()
    # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
    out_circles = []
    
    # Sort the accumulator based on the votes for the candidate circles 
    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold: 
            # Shortlist the circle for final result
            out_circles.append((x, y, r, current_vote_percentage))
    
    # Post process the results, can add more post processing later.
    if post_process :
        out_circles = post_procces(out_circles, pixel_threshold)
        
    # Draw shortlisted circles on the output image
    for x, y, r, v in out_circles:
        if circle_color =='Red':
            output_img = cv2.circle(output_img, (x,y), r, (255,0,0), 2)
        elif circle_color =='Blue':
           output_img = cv2.circle(output_img, (x,y), r, (0,0,205), 2)
        elif circle_color == 'Green':
            output_img = cv2.circle(output_img, (x,y), r, (50,205,50), 2)
    
    return output_img



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

###################################################################### HOUGH ELIPSES ####################################################################
    
def draw_hough_elipses(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(512,512))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret , threshold = cv2.threshold(imgray,20,255,0)
    edges = cv2.Canny(imgray,100, 200)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    img= cv2.drawContours(img,contours,-1,(0,255,0),5)
    return img


# Note this function comlixty is to high O(n^4) so it does not work efficiently
def hough_ellipse(image, threshold=200):
    edges = cv2.Canny(image, 5, 200)
    height, width = edges.shape
    accumulator = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            if edges[y][x] == 255:
                for a in range(1, width//2):
                    b = int(a * np.sqrt((x - width/2)*2 + (y - height/2)*2) / max(width/2, height/2))
                    if b > 0:
                        for theta in range(0, 360):
                            theta_rad = np.deg2rad(theta)
                            x0 = int(x - a * np.cos(theta_rad))
                            y0 = int(y + b * np.sin(theta_rad))
                            if x0 >= 0 and x0 < width and y0 >= 0 and y0 < height:
                                accumulator[y0][x0] += 1
    
    ellipses = []
    for y in range(height):
        for x in range(width):
            if accumulator[y][x] > threshold:
                ellipses.append((x,y))
    
    return ellipses


# def hough_ellipse(self):

#     # Canny Edge Detection using OpenCV
#     edges = cv2.Canny(self.gray_img, 50, 150)

#     # Find edge points
#     edge_points = np.transpose(np.nonzero(edges))

#     # Get the shape of the edges image
#     edges_shape = np.shape(edges)

#     # Initialize accumulator array
#     accumulator = np.zeros(edges_shape, dtype=np.uint)

#     # Define parameter space
#     a_max = max(edges_shape) // 2
#     a_range = np.arange(-a_max, a_max + 1)
#     b_max = min(edges_shape) // 2
#     b_range = np.arange(-b_max, b_max + 1)

#     # Iterate through edge points and vote for potential ellipses
#     for x, y in edge_points:
#         for a in a_range:
#             for b in b_range:
#                 if a != 0 and b != 0:  # Check if a and b are non-zero
#                     new_x = x + a
#                     if new_x < 0 or new_x >= edges_shape[0]:
#                         continue

#                     new_y = y + b
#                     if new_y < 0 or new_y >= edges_shape[1]:
#                         continue

#                     accumulator[new_x, new_y] += 1

#     # Threshold accumulator to find potential ellipses
#     threshold = 100  
#     ellipses = np.argwhere(accumulator > threshold)

#     # Convert to ellipse parameters (center and radii)
#     detected_ellipses = []
#     for x, y in ellipses:
#         detected_ellipses.append((x, y, a_range[y], b_range[x]))

#     # Superimpose detected ellipses on the original image
#     result_image = QPixmap(self.input_image)
#     painter = QPainter(result_image)
#     painter.setPen(QColor(0, 255, 0))

#     for ellipse in detected_ellipses:
#         painter.drawEllipse(*ellipse)
#     painter.end()

#     self.ui.hough_output.setPixmap(result_image)

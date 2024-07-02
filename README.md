# Edge and Boundary Detection (Hough Transform and SNAKE)

## Assignment 2


### Overview
This project involves edge and boundary detection using various computer vision techniques such as the Canny edge detector and Hough Transform for detecting lines, circles, and ellipses, as well as the Active Contour Model (SNAKE) for contour evolution.

### Tasks Implemented

#### A) Tasks to Implement

1. **Edge Detection and Shape Detection**
    - **Canny Edge Detection:**
        - Applied Gaussian filter to smooth the image and remove noise.
        - Computed image gradient to identify edges.
        - Non-maximum suppression to thin the edges.
        - Edge tracking using hysteresis.
        
      ![Canny Edge Detection](![image](https://github.com/IbrahimEmad11/CompVision_Task2/assets/110200613/b958a85d-832f-46fa-8a8c-f9d4ff0c2753)
)
      
    - **Hough Transform for Line Detection:**
        - Line detection using parametric representation in θ and ρ space.
        
      ![Hough Line Detection](https://i.imgur.com/XtNH9Hp.png)
      
    - **Hough Transform for Circle Detection:**
        - Circle detection with a 3-dimensional parameter space for center coordinates and radius.
        
      ![Hough Circle Detection](https://i.imgur.com/6G6aeAs.png)
      ![Hough Circle Detection](https://i.imgur.com/X7BpmQN.png)
      
    - **Hough Transform for Ellipse Detection:**
        - Ellipse detection with a 5-dimensional parameter space for center coordinates, major and minor axes' lengths, and orientation angle.
        
      ![Hough Ellipse Detection](https://i.imgur.com/LKlhMx0.png)

2. **Active Contour Model (SNAKE)**
    - **Initialize Contour:**
        - Initialized the contour for segmentation using circle, rectangle, or points.
    - **Energy Functional:**
        - Calculated the energy for the given image and contour using image term, contour term, and curvature term.
    - **Gradient Functional:**
        - Calculated the gradient of the energy with respect to the contour.
    - **Evolve Contour:**
        - Updated the contour using a greedy algorithm that minimizes the energy functional.
    - **Freeman Chain Code:**
        - Converted the contour points into chain codes using the Freeman chain code algorithm.
        
      ![Active Contour](https://i.imgur.com/40xYp7n.png)

#### B) Reporting

- Report includes:
    - Detailed explanation of the methods used.
    - Code implementation.
    - Results with superimposed images.
    - Calculations of perimeter and area for the contours.

### Files Included

- `edge_detection.py`: Python script for detecting edges using the Canny edge detector.
- `hough_transform.py`: Python script for detecting lines, circles, and ellipses using Hough Transform.
- `snake_algorithm.py`: Python script for implementing the Active Contour Model (SNAKE).
- `results/`: Directory containing the images with superimposed detected shapes and contours.
- `report.pdf`: Detailed report including methodology, code, results, and calculations.

### Instructions to Run the Code

1. **Requirements:**
    - Python 3.x
    - NumPy
    - OpenCV
    - Matplotlib
    - Scikit-image

2. **Setup:**
    - Ensure all required libraries are installed.
    - Place the given images in the `images/` directory.

3. **Running the Scripts:**
    - To run the program for edges and shapes and apply the Active Contour Model:
      ```bash
      python task3.py
      ```
### Team Members
- **Ibrahim Emad** 
- **Ahmed Khaled**
- **Arsany Maged** 
- **Omar Atef** 
- **Kyrollos Emad** 

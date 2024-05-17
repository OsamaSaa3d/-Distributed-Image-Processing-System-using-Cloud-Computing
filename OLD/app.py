import cv2
import numpy as np
import io
from flask import Flask, request, send_file
from flask_cors import CORS
from mpi4py import MPI
import math
import time

app = Flask(__name__)
CORS(app)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def split_image(image, num_parts):
    # Slice image into (size) parts
    image_parts = np.array_split(image, size, axis=0)

    # Return back to original binary format, as received.
    for idx, slice in enumerate(image_parts):
        retval, buffer = cv2.imencode('.jpg', slice)
        image_parts[idx] = buffer.tobytes()

    return image_parts

def combine_image(parts):
    combined_image = np.concatenate(parts, axis=0)
    return combined_image

@app.route('/process_image', methods=['POST'])
def process_image():
    if rank == 0:  # Master node
        print("Master node receiving image and operation...")
        # Receive image and operation from client
        image_data = request.files['image'].read()
        operation = request.form['operation']

        # Convert binary image data to NumPy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode NumPy array to image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        print("Splitting image into parts...")
        image_parts = split_image(image, size)
        
        print("Distributing tasks to worker nodes...")
        # Distribute tasks to worker nodes
        for i in range(1, size):
            print("Sending task to worker node " + str(i))
            comm.send((operation, image_parts[i]), dest=i)

        print("Processing master's own task...")
        # Process the master's own task
        processed_image_data = process_image_task(operation, image_parts[0])

        print("Collecting processed images from worker nodes...")
        # Collect processed images from worker nodes
        processed_image_parts = []
        processed_image_parts.append(processed_image_data)

        for i in range(1, size):
            processed_image_part = comm.recv(source=i)
            processed_image_parts.append(processed_image_part)

        # Concatenate processed image parts along the vertical axis
        processed_image_data = combine_image(processed_image_parts)

        print("Sending processed image data back to client...")
        _, processed_image_data = cv2.imencode('.jpg', processed_image_data)

        # Return processed image data
        return send_file(
            io.BytesIO(processed_image_data),
            mimetype='image/jpeg'
        )

def process_image_task(operation, image_data):
    # Process image using OpenCV
    nparr = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Perform image processing operation based on 'operation'
    if operation == 'grayscale':
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif operation == 'blur':
        processed_image = cv2.GaussianBlur(image, (25, 25), 0)
    elif operation == 'color_inversion':
        processed_image = cv2.bitwise_not(image)
    elif operation == 'edge_detection':
        processed_image = cv2.Canny(image, 100, 200)
    elif operation == 'histogram_equalization':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.equalizeHist(gray_image)
    elif operation == 'sharpening':
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        processed_image = cv2.filter2D(image, -1, kernel)
    elif operation == 'thresholding':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    elif operation == 'dilation':
        kernel = np.ones((25, 25), np.uint8)
        processed_image = cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erosion':
        kernel = np.ones((25, 25), np.uint8)
        processed_image = cv2.erode(image, kernel, iterations=1)
    elif operation == 'opening':
        kernel = np.ones((25, 25), np.uint8)
        processed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        kernel = np.ones((25, 25), np.uint8)
        processed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'contour_detection':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    elif operation == 'skeletonization':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        processed_image = cv2.ximgproc.thinning(processed_image)
    elif operation == 'distance_transform':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        processed_image = cv2.distanceTransform(processed_image, cv2.DIST_L2, 5)
    elif operation == 'connected_component_analysis':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_image, connectivity=8)
        processed_image = cv2.cvtColor(labels.astype(np.uint8) * (255 / num_labels), cv2.COLOR_GRAY2BGR)
    elif operation == 'blob_detection':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(processed_image)
        processed_image = cv2.drawKeypoints(image.copy(), keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif operation == 'hough_transform':
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        processed_image = image
    else:
        processed_image = image

    # Return processed image data
    return processed_image

if __name__ == '__main__':
    if rank == 0:
        print("Master node starting Flask application...")
        app.run(host='0.0.0.0', port=5000)
    else:  # Worker nodes
        while True:
            image_data = None
            while True:
            # Receive task from master node
                operation, image_data = comm.recv(source=0)

                # Check if the received task is a termination signal
                if image_data is not None:
                    break  # Break the loop if termination signal is received

            print(f"Worker node {rank} receiving task...")
            # Process image task
            processed_image_data = process_image_task(operation, image_data)

            # Send processed image back to master node
            comm.send(processed_image_data, dest=0)

            # Print termination message after the loop
            print(f"Worker node {rank} has terminated.")

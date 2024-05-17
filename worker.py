#worker code
import cv2
import numpy as np
import io
import time
from flask import Flask, request, send_file
from flask_cors import CORS
from mpi4py import MPI
import boto3
import json
import uuid

rank = 1

# Initialize S3 and SQS clients
s3 = boto3.client('s3', region_name='eu-north-1')
sqs = boto3.client('sqs', region_name='eu-north-1')
bucket_name = 'distproject-my-image-processing-bucket'
task_queue_url = 'https://sqs.eu-north-1.amazonaws.com/654654179409/image_pieces_queue.fifo'
response_queue_url = 'https://sqs.eu-north-1.amazonaws.com/654654179409/processed_image_queue.fifo'

def upload_to_s3(image_data, key):
    print(f"Uploading to S3 with key: {key}")
    s3.put_object(Bucket=bucket_name, Key=key, Body=image_data)
    return key

def download_from_s3(key):
    print(f"Downloading from S3 with key: {key}")
    time.sleep(2)  # Adding delay to ensure the object is available in S3
    response = s3.get_object(Bucket=bucket_name, Key=key)
    return response['Body'].read()

def send_message_to_sqs(queue_url, s3_key, operation, group_id, deduplication_id):
    message = {
        's3_key': s3_key,
        'operation': operation
    }
    print(f"Sending message to SQS: {message}")
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message),
        MessageGroupId=group_id,
        MessageDeduplicationId=deduplication_id
    )

def receive_message_from_sqs(queue_url):
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,  # Receive 1 message at a time
        WaitTimeSeconds=10,
        VisibilityTimeout=10  # Visibility timeout of 10 seconds
    )
    return response.get('Messages', [])

def delete_message_from_sqs(queue_url, receipt_handle):
    sqs.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle
    )

def process_image_task(operation, s3_key):
    print(f"Processing image from S3 with key: {s3_key}")
    image_data = download_from_s3(s3_key)
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
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

    key = f'processed_image_{uuid.uuid4()}.jpg'
    return upload_to_s3(cv2.imencode('.jpg', processed_image)[1].tobytes(), key)

if __name__ == '__main__':
    while True:
        messages = receive_message_from_sqs(task_queue_url)
        if messages:
            message = messages[0]
            body = json.loads(message['Body'])
            s3_key = body['s3_key']
            operation = body['operation']
            
            if body.get('terminate', False):
                print("Termination signal received. Shutting down worker.")
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                break

            print(f"Worker node {rank} processing task...")
            try:
                processed_image_url = process_image_task(operation, s3_key)
                group_id = 'processed_image_group'
                deduplication_id = f'processed_image_{rank}_{uuid.uuid4()}'
                send_message_to_sqs(response_queue_url, processed_image_url, operation, group_id, deduplication_id)
                delete_message_from_sqs(task_queue_url, message['ReceiptHandle'])
            except Exception as e:
                print(f"Error processing task: {e}")
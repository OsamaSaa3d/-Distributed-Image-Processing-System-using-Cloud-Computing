#master code
import cv2
import numpy as np
import io
import sys
import time
from flask import Flask, request, send_file
from flask_cors import CORS
from mpi4py import MPI
import boto3
import json
import uuid
import signal

app = Flask(__name__)
CORS(app)

# Initialize S3 and SQS clients
s3 = boto3.client('s3', region_name='eu-north-1')
sqs = boto3.client('sqs', region_name='eu-north-1')
bucket_name = 'distproject-my-image-processing-bucket'
task_queue_url = 'https://sqs.eu-north-1.amazonaws.com/654654179409/image_pieces_queue.fifo'
response_queue_url = 'https://sqs.eu-north-1.amazonaws.com/654654179409/processed_image_queue.fifo'

size = 4

def signal_handler(sig, frame):
    print('Termination signal received in master. Shutting down...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def split_image(image, num_parts):
    image_parts = np.array_split(image, num_parts, axis=0)
    for idx, slice in enumerate(image_parts):
        retval, buffer = cv2.imencode('.jpg', slice)
        image_parts[idx] = buffer.tobytes()
    return image_parts

def combine_image(parts):
    combined_image = np.concatenate(
        [cv2.imdecode(np.frombuffer(part, np.uint8), cv2.IMREAD_COLOR) for part in parts],
        axis=0
    )
    return combined_image

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

@app.route('/process_image', methods=['POST'])
def process_image():

    print("Master node receiving image and operation...")
    image_data = request.files['image'].read()
    operation = request.form['operation']

    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print("Splitting image into parts...")
    image_parts = split_image(image, size)

    print("Distributing tasks to worker nodes...")
    for idx, image_part in enumerate(image_parts):
        key = f'image_part_{uuid.uuid4()}.jpg'
        upload_to_s3(image_part, key)
        group_id = 'image_processing_group'
        deduplication_id = f'image_part_{idx}_{uuid.uuid4()}'
        send_message_to_sqs(task_queue_url, key, operation, group_id, deduplication_id)

    processed_image_parts = []
    for _ in range(size):
        while True:
            messages = receive_message_from_sqs(response_queue_url)
            if messages:
                message = messages[0]
                body = json.loads(message['Body'])
                s3_key = body['s3_key']
                print(f"Downloading processed image part from S3 with key: {s3_key}")
                processed_image_part = download_from_s3(s3_key)
                processed_image_parts.append(processed_image_part)
                delete_message_from_sqs(response_queue_url, message['ReceiptHandle'])
                break

    print("Combining processed images...")
    combined_image = combine_image(processed_image_parts)

    retval, buffer = cv2.imencode('.jpg', combined_image)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == '__main__':
    print("Master node starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
    print("Terminating... please wait")
    time.sleep(2)
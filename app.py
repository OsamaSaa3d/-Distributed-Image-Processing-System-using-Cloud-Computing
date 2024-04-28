from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route('/process_image', methods=['POST'])
def process_image():
    # Receive image and operation from client
    image_data = request.files['image'].read()
    operation = request.form['operation']

    # Process image using OpenCV
    nparr = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Perform image processing operation based on 'operation'
    if operation == 'grayscale':
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif operation == 'blur':
        processed_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif operation == 'color_inversion':
        processed_image = cv2.bitwise_not(image)
    else:
        processed_image = image

    # Encode processed image to JPEG format
    _, processed_image_data = cv2.imencode('.jpg', processed_image)

    # Return processed image data
    return send_file(
        io.BytesIO(processed_image_data),
        mimetype='image/jpeg'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

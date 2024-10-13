import os
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import cv2 as cv
import numpy as np
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# CORS Middleware (Handled automatically by Vercel, so you can skip it)
# from flask_cors import CORS
# CORS(app)

# Load models (Mock the model load)
from scoliovis.kprcnn import predict, kprcnn_to_scoliovis_api_format  # KPRCNN

@app.route("/")
def read_root():
    return jsonify({
        "Hello": "World",
        "Message": "Welcome to ScolioVis-API! Send a POST request these APIs to get started!",
        "ModelPredict": "/v2/getprediction"
    })

@app.route('/v2/getprediction', methods=['POST'])
def get_prediction_v2():
    # Get the image from the request
    file = request.files['image']
    image = Image.open(BytesIO(file.read())).convert('RGB')
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)

    # Keypoint RCNN Predict
    bboxes, keypoints, scores = predict(image)[0]
    api_object = kprcnn_to_scoliovis_api_format(bboxes, keypoints, scores, image.shape)

    return jsonify(api_object)

if __name__ == "__main__":
    app.run()

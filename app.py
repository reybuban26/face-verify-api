from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

# Suppress TensorFlow logs to keep logs clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace

app = Flask(__name__)

def process_image(file_stream):
    # Convert uploaded file to numpy array directly
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

@app.route('/', methods=['GET'])
def home():
    return "Face Verification API (DeepFace) is Running!"

@app.route('/verify', methods=['POST'])
def verify():
    try:
        if 'id_image' not in request.files or 'selfie_image' not in request.files:
            return jsonify({"error": "Missing images"}), 400

        # 1. Load Images from memory
        id_img = process_image(request.files['id_image'])
        selfie_img = process_image(request.files['selfie_image'])

        # 2. DeepFace Verify
        # Gamit ang "VGG-Face" model at "opencv" detector (magaan at mabilis)
        result = DeepFace.verify(
            img1_path = id_img,
            img2_path = selfie_img,
            model_name = "VGG-Face",
            detector_backend = "opencv",
            enforce_detection = False, 
            align = True
        )

        # 3. Process Result
        is_match = result['verified']
        distance = result['distance']
        
        # Convert distance to confidence score (Estimate)
        # VGG-Face threshold is usually around 0.40
        confidence = 0
        if is_match:
            confidence = max(0, min(100, (1 - distance) * 100 + 20)) 
        else:
            confidence = max(0, (1 - distance) * 100)

        return jsonify({
            "match": bool(is_match),
            "confidence": round(confidence, 2),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
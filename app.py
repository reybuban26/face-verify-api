from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import gc # Import Garbage Collector para maglinis ng RAM

# Suppress TensorFlow logs
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
    return "Face Verification API (ArcFace Lite) is Running!"

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Force Clean Memory bago mag-start
        gc.collect()

        if 'id_image' not in request.files or 'selfie_image' not in request.files:
            return jsonify({"error": "Missing images"}), 400

        # 1. Load Images
        id_img = process_image(request.files['id_image'])
        selfie_img = process_image(request.files['selfie_image'])

        # 2. DeepFace Verify using ARCFACE
        # ArcFace is minimal (~23MB weights) and fits in 512MB RAM
        result = DeepFace.verify(
            img1_path = id_img,
            img2_path = selfie_img,
            model_name = "ArcFace",  # <--- PINALITAN NATIN NITO
            detector_backend = "opencv",
            enforce_detection = False, 
            align = True
        )

        # 3. Process Result
        is_match = result['verified']
        distance = result['distance']
        
        # ArcFace Threshold is distinct. Usually around 0.68
        # We adjust the confidence calculation for ArcFace
        confidence = 0
        if is_match:
            confidence = max(0, min(100, (1 - distance) * 100 + 30)) 
        else:
            confidence = max(0, (1 - distance) * 100)

        # Clean up memory after verify
        del id_img
        del selfie_img
        del result
        gc.collect()

        return jsonify({
            "match": bool(is_match),
            "confidence": round(confidence, 2),
            "status": "success"
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
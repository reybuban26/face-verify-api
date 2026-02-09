from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import gc

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deepface import DeepFace

app = Flask(__name__)

def process_image(file_stream):
    # 1. Read Image from Request
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 2. RESIZE IMAGE (CRITICAL STEP FOR FREE TIER)
    # Phone images are too big (4000px+). We resize to max 600px width.
    # This reduces RAM usage from ~100MB to ~5MB per image.
    height, width = img.shape[:2]
    max_width = 600
    
    if width > max_width:
        scaling_factor = max_width / float(width)
        new_height = int(height * scaling_factor)
        img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)
    
    return img

@app.route('/', methods=['GET'])
def home():
    return "Face Verification API (Optimized ArcFace) is Running!"

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Clean memory before starting
        gc.collect()

        if 'id_image' not in request.files or 'selfie_image' not in request.files:
            return jsonify({"error": "Missing images"}), 400

        # Load and Resize Images
        id_img = process_image(request.files['id_image'])
        selfie_img = process_image(request.files['selfie_image'])

        # DeepFace Verify using ARCFACE
        result = DeepFace.verify(
            img1_path = id_img,
            img2_path = selfie_img,
            model_name = "ArcFace",
            detector_backend = "opencv",
            enforce_detection = False, 
            align = True
        )

        # Process Result
        is_match = result['verified']
        distance = result['distance']
        
        # ArcFace Confidence Calculation
        confidence = 0
        if is_match:
            confidence = max(0, min(100, (1 - distance) * 100 + 30)) 
        else:
            confidence = max(0, (1 - distance) * 100)

        # Force Memory Cleanup
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
from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np

app = Flask(__name__)

def process_image(file_stream):
    # Convert uploaded file to numpy array directly (no saving to disk needed)
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img

@app.route('/', methods=['GET'])
def home():
    return "Face Verification API is Running!"

@app.route('/verify', methods=['POST'])
def verify():
    try:
        if 'id_image' not in request.files or 'selfie_image' not in request.files:
            return jsonify({"error": "Missing images"}), 400

        # 1. Load Images directly from memory
        id_img = process_image(request.files['id_image'])
        selfie_img = process_image(request.files['selfie_image'])

        # 2. Find Faces (Upsampling for better detection)
        try:
            id_encodings = face_recognition.face_encodings(id_img)
            if not id_encodings:
                return jsonify({"match": False, "confidence": 0, "error": "No face in ID"}), 200
            id_encode = id_encodings[0]
        except Exception as e:
            return jsonify({"error": f"ID Processing Error: {str(e)}"}), 500

        try:
            selfie_encodings = face_recognition.face_encodings(selfie_img)
            if not selfie_encodings:
                return jsonify({"match": False, "confidence": 0, "error": "No face in Selfie"}), 200
            selfie_encode = selfie_encodings[0]
        except Exception as e:
            return jsonify({"error": f"Selfie Processing Error: {str(e)}"}), 500

        # 3. Compare
        # Tolerance: 0.5 (Strict), 0.6 (Standard). Lower is stricter.
        match_result = face_recognition.compare_faces([id_encode], selfie_encode, tolerance=0.5)
        face_distance = face_recognition.face_distance([id_encode], selfie_encode)[0]
        
        # Convert distance to confidence score (0 to 100)
        score = round((1 - face_distance) * 100, 2)
        is_match = bool(match_result[0])

        return jsonify({
            "match": is_match,
            "confidence": score,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
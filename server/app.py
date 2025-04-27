"""
Author: Rana Abdellatif
Backend Flask API for Encrypted Sign Language Translation Project
"""

# === app.py ===
import os
import uuid
import shutil
import cv2
import mediapipe as mp
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

#initializing app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') #my secret key
CORS(app)

#encryption key (securely stored server-side) - AES256
encryption_key = get_random_bytes(32) # <- must be 32 bytes

#create necessary folders if dne
UPLOAD_FOLDER = 'uploads'
ENCRYPTED_FOLDER = 'encrypted'
DECRYPTED_FOLDER = 'decrypted'

for folder in [UPLOAD_FOLDER, ENCRYPTED_FOLDER, DECRYPTED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

#connect MongoDB
client = MongoClient('mongodb+srv://ranran645:8aAUvzPzrz3hzSuJ@cluster0.c15ynzr.mongodb.net/') #my unique string
#db initial setup
db = client['asl_database']
sessions_collection = db['sessions']

#load ASL model
model = tf.keras.models.load_model('mini_model.h5') 

#label map based on kaggle model
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

#helper -> encrypt file
def encrypt_file(input_path, output_path):
    cipher = AES.new(encryption_key, AES.MODE_EAX)
    with open(input_path, 'rb') as f:
        data = f.read()
    ciphertext, tag = cipher.encrypt_and_digest(data)
    with open(output_path, 'wb') as f:
        #replace with encrypted values
        [f.write(x) for x in (cipher.nonce, tag, ciphertext)]

#helper -> decrypt file
def decrypt_file(input_path):
    output_path = input_path.replace('encrypted', 'decrypted')
    with open(input_path, 'rb') as f:
        nonce, tag, ciphertext = [f.read(x) for x in (16, 16, -1)]
    cipher = AES.new(encryption_key, AES.MODE_EAX, nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    with open(output_path, 'wb') as f:
        f.write(data)
    return output_path

#func to predict the english letters based on image (vid) reading
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue  # Only predict every 5th frame (skip others)

        # Preprocess frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (100, 100))  # Same as training size
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)

        # Predict
        pred = model.predict(frame, verbose=0)
        pred_class_idx = np.argmax(pred, axis=1)[0]
        predictions.append(pred_class_idx)

    cap.release()

    if not predictions:
        return ""

    # Smarter voting: take top 3 most common predictions and pick the one with most confidence
    from collections import Counter
    top_classes = Counter(predictions).most_common(3)
    most_common_class_idx = top_classes[0][0]

    return class_names[most_common_class_idx]


#upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    video = request.files['video']
    filename = f"{uuid.uuid4().hex}.webm"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    encrypted_path = os.path.join(ENCRYPTED_FOLDER, filename)

    video.save(video_path)

    #encrypt vid immediately
    encrypt_file(video_path, encrypted_path)
    os.remove(video_path)

    #decrypt vid for processing
    decrypted_path = decrypt_file(encrypted_path)
    
    #store predition in var
    translation = predict_video(decrypted_path)

    #save sesh to mongodb
    session_data = {
        "translation": translation,
        "timestamp": datetime.utcnow()
    }
    sessions_collection.insert_one(session_data)

    #clean up
    os.remove(encrypted_path)
    os.remove(decrypted_path)

    return jsonify({"translation": translation})

#sessions route
@app.route('/sessions', methods=['GET'])
def get_sessions():
    sessions = list(sessions_collection.find({}, {'_id': 0}))
    return jsonify(sessions)

if __name__ == '__main__':
    app.run(debug=True)

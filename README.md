
# Encrypted ASL Recognition - Flask API

This is the backend API for the Encrypted Sign Language Recognition project. It handles encrypted video uploads, gesture recognition using a pre-trained ASL model, and session storage in MongoDB.

## 🔐 Key Features

- AES-256 encryption for all uploaded video files
- Flask-based REST API for video upload and processing
- Real-time gesture recognition with OpenCV + MediaPipe (or pre-trained model)
- MongoDB session storage via PyMongo
- Secure file handling and automatic cleanup

## ⚙️ Tech Stack

- Python 3.x
- Flask
- PyCryptodome (for AES encryption)
- OpenCV, MediaPipe
- MongoDB Atlas
- Flask-CORS

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MongoDB Atlas account (or local instance)

### Installation

```bash
git clone https://github.com/your-username/encrypted-asl-api.git
cd encrypted-asl-api
pip install -r requirements.txt


Running locally::

python /insert path/app.py

Open new env if trying to test out training model

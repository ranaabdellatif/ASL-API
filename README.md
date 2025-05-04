# Encrypted-ASL-API
Sadly, current sign language recognition models do not encrypt gesture data, making them vulnerable to interception, modification, and man-in-the-middle attacks during transmission.
This is a secure API that applies end-to-end encryption (E2EE) to sign language data before transmission.

# How to run

Install all necessary Python packages with requirements.txt





Running app.py:

python /Users/ranaabdellatif/Documents/GitHub/Encrypted-ASL-API/app.py

Running detection.py:
python -m venv new_venv
source new_venv/bin/activate
pip install opencv-python mediapipe
python /Users/ranaabdellatif/Documents/GitHub/Encrypted-ASL-API/detect_and_send.py


python /Users/ranaabdellatif/Documents/GitHub/Encrypted-ASL-API/detection.py

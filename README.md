# AIMS Drone Project

## Gesture-Controlled Drone Navigation & Safe Landing

This project explores controlling a drone using hand gestures and making landings safer using computer vision.  
We experimented with two different approaches for gesture detection and combined them with a landing surface safety check.

---

## What this project does

- Control drone movement using hand gestures (Up, Down, Left, Right)
- Detect landing intent using a downward gesture
- Check whether the surface below is safe to land on using camera input
- Display everything live using Streamlit

---

## Files in this repository

### app.py – CNN-based Gesture Control (Experimental)

- Uses a CNN model trained on our own dataset
- Runs on Streamlit with live camera input
- Predicts directions like up, down, left, right
- This version was mainly used to understand gesture recognition using deep learning

---

### app2.py – MediaPipe Gesture Control + Safe Landing (Main Version)

- Uses MediaPipe Hands for real-time hand tracking
- Converts hand movements into directions using simple geometry
- Down gesture automatically triggers landing safety check
- Uses a phone camera (IP webcam) as a downward-facing camera
- Analyzes the landing surface using:
  - edge detection
  - texture variation
  - brightness
- Displays SAFE TO LAND / UNSAFE LANDING ZONE live

This is the final and most reliable version of the project.

---

### collect_data.py

- Used to collect gesture images for training the CNN
- Saves images into class-wise folders (up, down, left, right)
- Only required for the CNN approach

---

### train_detection_model.py

- Trains the CNN on the collected dataset
- Generates the trained model file
- Used only for the experimental approach

---

### direction_model.h5

- Trained CNN model
- Loaded by app.py for gesture prediction

---

## Why two approaches?

- The CNN approach helped us understand dataset creation and model training
- The MediaPipe approach turned out to be faster, more accurate, and easier to run live

Because of this, MediaPipe was used for the main implementation.

---

## Technologies Used

- Python
- Streamlit
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- IP Webcam (mobile camera)



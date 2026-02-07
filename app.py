import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model


MODEL_PATH = "direction_model.h5"
IMG_SIZE = 128
CONF_THRESHOLD = 0.6  


labels = ["down", "left", "right", "up"]


model = load_model(MODEL_PATH)


st.set_page_config(page_title="Hand Direction Detection", layout="centered")
st.title("Hand Gesture Direction Detection")

st.write(
    "Place your hand inside the green box and show **UP / DOWN / LEFT / RIGHT**.\n"
    "If no hand is detected, the system will block the command."
)

run = st.checkbox("Enable Camera")

frame_window = st.image([])


def preprocess(roi):
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi


cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    
    x1, y1, x2, y2 = 300, 100, 600, 400
    roi = frame[y1:y2, x1:x2]

   
    img = preprocess(roi)
    preds = model.predict(img, verbose=0)[0]
    confidence = float(np.max(preds))

    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    if confidence < CONF_THRESHOLD:
        # NO HAND / LOW CONFIDENCE
        cv2.putText(
            frame,
            "CANNOT CAPTURE COMMAND",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        frame_window.image(frame, channels="BGR")

        st.subheader("Detected Direction")
        st.error("Cannot capture command")
        st.progress(confidence)

    else:
        # VALID GESTURE
        gesture = labels[np.argmax(preds)]

        cv2.putText(
            frame,
            f"{gesture.upper()} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        frame_window.image(frame, channels="BGR")

        st.subheader("Detected Direction")
        st.success(gesture.upper())
        st.progress(confidence)


cap.release()

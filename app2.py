import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.python.solutions import hands as mp_hands


st.set_page_config(page_title="Gesture-Controlled Safe Drone Landing", layout="centered")
st.title("Gesture-Controlled Drone with Automatic Safe Landing")

st.write(
    "• Laptop camera → hand gesture navigation\n"
    "• Phone camera → landing surface inspection\n"
    "• Showing **DOWN** automatically checks landing safety"
)

run = st.checkbox("Enable Camera")

frame_window = st.image([])
status_box = st.empty()
metrics_box = st.empty()


if "last_landing_time" not in st.session_state:
    st.session_state.last_landing_time = 0  # cooldown

LANDING_COOLDOWN = 2.0  # seconds


hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)


if "gesture_cam" not in st.session_state:
    st.session_state.gesture_cam = cv2.VideoCapture(0)

gesture_cam = st.session_state.gesture_cam


def get_direction(landmarks):
    wrist = landmarks[0]
    index_tip = landmarks[8]

    dx = index_tip.x - wrist.x
    dy = index_tip.y - wrist.y

    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"


def assess_landing_surface(frame):
    h, w, _ = frame.shape
    roi = frame[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    texture_variance = np.var(gray)
    bright_ratio = np.sum(gray > 200) / gray.size

    SAFE = (
        edge_density < 0.05 and
        texture_variance < 500 and
        bright_ratio < 0.10
    )

    return SAFE, edge_density, texture_variance, bright_ratio

if run:
    while run:
        ret, frame = gesture_cam.read()
        if not ret:
            st.error("Gesture camera not working")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb)

        current_time = time.time()

       
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            direction = get_direction(hand_landmarks.landmark)

            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"DIRECTION: {direction}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            
            if direction == "DOWN" and current_time - st.session_state.last_landing_time > LANDING_COOLDOWN:
                st.session_state.last_landing_time = current_time
                status_box.warning("DOWN detected — checking landing surface")

                PHONE_CAM_URL = "http://100.125.178.65:8080/video"

                landing_cam = cv2.VideoCapture(PHONE_CAM_URL)
                time.sleep(0.5)  # allow stream to initialize
                ret2, land_frame = landing_cam.read()
                landing_cam.release()

                if not ret2:
                    status_box.error("Landing camera not reachable")
                else:
                    SAFE, edge_d, tex_v, bright_r = assess_landing_surface(land_frame)

                    if SAFE:
                        status_box.success("SAFE TO LAND")
                    else:
                        status_box.error("UNSAFE LANDING ZONE")

                    metrics_box.info(
                        f"""
                        **Landing Surface Metrics**
                        - Edge density: `{edge_d:.3f}`
                        - Texture variance: `{tex_v:.1f}`
                        - Brightness ratio: `{bright_r:.2f}`
                        """
                    )

                    frame = land_frame  # show phone feed

            else:
                if direction != "DOWN":
                    status_box.info(f"Moving {direction}")
                    metrics_box.empty()

        else:
            status_box.error("Cannot capture command")

        frame_window.image(frame, channels="BGR")
        time.sleep(0.03)

else:
    st.info("Camera off")












import cv2
import os

# ============================
# CONFIGURATION
# ============================

BASE_PATH = r"C:\Users\lenovo\OneDrive\Pictures\Desktop\dataset"
CATEGORY = "direction"
LABEL = "right"

SAVE_DIR = os.path.join(BASE_PATH, CATEGORY, LABEL)
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================
# CONTINUE COUNT (NO OVERWRITE)
# ============================

existing_images = [
    f for f in os.listdir(SAVE_DIR)
    if f.endswith(".jpg") or f.endswith(".png")
]
count = len(existing_images)

print(f"Starting capture from image index: {count}")

# ============================
# CAMERA SETUP
# ============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot access camera")
    exit()

print("====================================")
print("DATASET CAPTURE MODE: DIRECTION-RIGHT")
print("====================================")
print(" - Put ONLY your hand inside the green box")
print(" - Fingers pointing RIGHT")
print(" - Press 'c' to capture")
print(" - Press 'q' to quit")
print("------------------------------------")

# ============================
# MAIN LOOP
# ============================

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame")
        break

    frame = cv2.flip(frame, 1)

    # ROI
    x1, y1, x2, y2 = 300, 100, 600, 400
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "CAPTURING: RIGHT",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Hand ROI", roi)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        img_path = os.path.join(SAVE_DIR, f"{count}.jpg")
        cv2.imwrite(img_path, roi)
        print(f"Saved image: {count}.jpg")
        count += 1

    elif key == ord('q'):
        print("Exiting capture mode...")
        break

# ============================
# CLEANUP
# ============================

cap.release()
cv2.destroyAllWindows()

print(f"Total images now in RIGHT folder: {count}")
print("RIGHT capture completed.")


import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import keyboard
import pyautogui
import warnings
import os
import logging
from plyer import notification

# Remove warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('mediapipe').setLevel(logging.CRITICAL)

# Mediapipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

# Camera variables
cap = None
camera_on = False

# Blink variables
last_blink_time = time.time()

# Zoom variables
face_width_history = []
history_size = 7

previous_face_width = None
movement_threshold = 40

current_zoom = 100
min_zoom = 80
max_zoom = 200
zoom_step = 10


# Eye aspect ratio
def calculate_EAR(eye):

    v1 = np.linalg.norm(eye[1]-eye[5])
    v2 = np.linalg.norm(eye[2]-eye[4])
    h = np.linalg.norm(eye[0]-eye[3])

    return (v1+v2)/(2.0*h)


# System Blink Alert
def blink_alert():

    notification.notify(
        title="Eye Monitor",
        message="Please blink your eyes!",
        timeout=3
    )


# Toggle camera
def toggle_camera():

    global camera_on
    global cap

    camera_on = not camera_on

    if camera_on:
        cap = cv2.VideoCapture(0)
    else:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


keyboard.add_hotkey("ctrl+shift+e", toggle_camera)


# Zoom functions
def zoom_in():

    pyautogui.keyDown("ctrl")
    pyautogui.scroll(400)
    pyautogui.keyUp("ctrl")


def zoom_out():

    pyautogui.keyDown("ctrl")
    pyautogui.scroll(-400)
    pyautogui.keyUp("ctrl")


# Camera thread
def run_camera():

    global last_blink_time
    global cap
    global previous_face_width
    global current_zoom
    global face_width_history

    while True:

        if not camera_on:
            time.sleep(0.2)
            continue

        if cap is None:
            time.sleep(0.2)
            continue

        ret, frame = cap.read()

        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                h, w, _ = frame.shape

                left_eye_idx = [33,160,158,133,153,144]

                eye_points = []

                for idx in left_eye_idx:

                    x = int(face_landmarks.landmark[idx].x*w)
                    y = int(face_landmarks.landmark[idx].y*h)

                    eye_points.append(np.array([x,y]))

                ear = calculate_EAR(eye_points)

                if ear < 0.25:
                    last_blink_time = time.time()

                p1 = face_landmarks.landmark[234]
                p2 = face_landmarks.landmark[454]

                x1,y1 = int(p1.x*w), int(p1.y*h)
                x2,y2 = int(p2.x*w), int(p2.y*h)

                face_width = np.linalg.norm(
                    np.array([x1,y1]) - np.array([x2,y2])
                )

                face_width_history.append(face_width)

                if len(face_width_history) > history_size:
                    face_width_history.pop(0)

                smooth_width = np.mean(face_width_history)

                if previous_face_width is None:
                    previous_face_width = smooth_width

                change = smooth_width - previous_face_width

                if abs(change) > movement_threshold:

                    # FAR → Zoom In
                    if change < 0 and current_zoom < max_zoom:

                        zoom_in()

                        current_zoom += zoom_step
                        previous_face_width = smooth_width
                        time.sleep(1)

                    # CLOSE → Zoom Out
                    elif change > 0 and current_zoom > min_zoom:

                        zoom_out()

                        current_zoom -= zoom_step
                        previous_face_width = smooth_width
                        time.sleep(1)

                # Blink alert after 10 seconds
                if time.time() - last_blink_time > 10:

                    blink_alert()
                    last_blink_time = time.time()

        cv2.imshow("Eye Monitor Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()


threading.Thread(target=run_camera, daemon=True).start()

print("Press CTRL + SHIFT + E to start or stop Eye Monitor")

while True:
    time.sleep(1)
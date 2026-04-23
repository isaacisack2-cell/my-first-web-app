import cv2
import pyautogui
import math
import time
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


CAMERA_ID       = 0
SMOOTHING       = 7
CLICK_THRESHOLD = 30
FRAME_REDUCTION = 80
CLICK_COOLDOWN  = 0.4
MODEL_PATH      = "hand_landmarker.task"
MODEL_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()
prev_x, prev_y = 0, 0


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Indownload model ya mkono... (mara moja tu ~5MB)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Model imedownload!")
    else:
        print("✅ Model ipo tayari.")


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def smooth(curr, prev, factor):
    return prev + (curr - prev) / factor


def get_lms_px(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def fingers_up(lms):
    tips   = [4,  8, 12, 16, 20]
    joints = [3,  6, 10, 14, 18]
    up = [lms[tips[0]][0] < lms[joints[0]][0]]
    for i in range(1, 5):
        up.append(lms[tips[i]][1] < lms[joints[i]][1])
    return up


def is_fist(lms):
    tips = [8, 12, 16, 20]
    mcps = [5,  9, 13, 17]
    return all(lms[t][1] > lms[m][1] for t, m in zip(tips, mcps))


def map_to_screen(x, y, w, h):
    active_w = w - 2 * FRAME_REDUCTION
    active_h = h - 2 * FRAME_REDUCTION
    sx = (x - FRAME_REDUCTION) / active_w * SCREEN_W
    sy = (y - FRAME_REDUCTION) / active_h * SCREEN_H
    return max(0, min(SCREEN_W, sx)), max(0, min(SCREEN_H, sy))


def draw_hand(frame, lms):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ]
    for a, b in connections:
        cv2.line(frame, lms[a], lms[b], (0, 200, 100), 2)
    for pt in lms:
        cv2.circle(frame, pt, 5, (255, 255, 255), cv2.FILLED)


def draw_overlay(frame, status, color, h, w):
    cv2.rectangle(frame, (0, 0), (320, 55), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"KIPANYA: {status}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    guide = [
        "Index tu   = Hamisha",
        "Pinch      = Click",
        "Middle tu  = Right Click",
        "Index+Mid  = Scroll",
        "Ngumi      = Drag",
        "Q          = Toka",
    ]
    for i, line in enumerate(guide):
        cv2.putText(frame, line, (10, h - 135 + i * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def main():
    global prev_x, prev_y

    download_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    dragging   = False
    last_click = 0

    print(f"🐭 KIPANYA v2 imeanza! Screen: {SCREEN_W}x{SCREEN_H}")
    print("   Bonyeza Q kutoka.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Kamera haifanyi kazi!")
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        cv2.rectangle(frame,
            (FRAME_REDUCTION, FRAME_REDUCTION),
            (w - FRAME_REDUCTION, h - FRAME_REDUCTION),
            (0, 255, 100), 2)

        status = "Hakuna mkono"
        color  = (100, 100, 100)

        if result.hand_landmarks:
            lms  = get_lms_px(result.hand_landmarks[0], w, h)
            draw_hand(frame, lms)

            up   = fingers_up(lms)
            fist = is_fist(lms)
            now  = time.time()

            ix, iy = lms[8]
            sx, sy = map_to_screen(ix, iy, w, h)
            sx = smooth(sx, prev_x, SMOOTHING)
            sy = smooth(sy, prev_y, SMOOTHING)
            prev_x, prev_y = sx, sy

            if fist:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                pyautogui.moveTo(sx, sy)
                status = "DRAG"
                color  = (0, 140, 255)
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

                if up[1] and not up[2]:
                    pyautogui.moveTo(sx, sy)
                    status = "HAMISHA"
                    color  = (0, 255, 100)

                pinch = dist(lms[8], lms[4])
                if pinch < CLICK_THRESHOLD:
                    if now - last_click > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click = now
                    status = "CLICK!"
                    color  = (0, 220, 255)
                elif up[2] and not up[1] and not up[3]:
                    if now - last_click > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        last_click = now
                    status = "RIGHT CLICK"
                    color  = (50, 50, 255)
                elif up[1] and up[2] and not up[3]:
                    if iy < h // 3:
                        pyautogui.scroll(3)
                        status = "SCROLL JUU"
                    elif iy > 2 * h // 3:
                        pyautogui.scroll(-3)
                        status = "SCROLL CHINI"
                    else:
                        status = "SCROLL..."
                    color = (255, 220, 0)

            cv2.circle(frame, (ix, iy), 13, color, cv2.FILLED)
            cv2.circle(frame, (ix, iy), 13, (255, 255, 255), 2)

        draw_overlay(frame, status, color, h, w)
        cv2.imshow("Kipanya v2 - Isaac @ ATC", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if dragging:
        pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("🐭 KIPANYA imesimama. Kwa heri!")


if __name__ == "__main__":
    main()

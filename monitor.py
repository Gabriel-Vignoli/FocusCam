import cv2
from ultralytics import YOLO
from plyer import notification
import pygame
import time

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.ogg")

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

last_alert = 0
show_alert_until = 0
frame_count = 0

button_top_left = (500, 10)
button_bottom_right = (630, 50)

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bx1, by1 = button_top_left
        bx2, by2 = button_bottom_right
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            pygame.mixer.stop()
            print("Music stopped by button.")

cv2.namedWindow("monitor")
cv2.setMouseCallback("monitor", mouse_click)

def draw_button(frame):
    cv2.rectangle(frame, button_top_left, button_bottom_right, (50, 50, 50), -1)
    cv2.putText(frame, "STOP", (button_top_left[0] + 10, button_top_left[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

while True:
    ret, frame = cap.read()

    if not ret or frame is None or frame.size == 0:
        continue

    frame_count += 1

    try:
        if frame_count % 3 != 0:
            if time.time() < show_alert_until:
                cv2.putText(frame, "LARGA O CELULAR!", (30, 240),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            draw_button(frame)
            cv2.imshow("monitor", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        results = model(frame, verbose=False, conf=0.3, imgsz=416)

        for r in results:
            for box in r.boxes:
                cls = model.names[int(box.cls)]
                conf = float(box.conf)
                if cls == "cell phone":
                    show_alert_until = time.time() + 3
                    if time.time() - last_alert > 30:
                        notification.notify(title="Foco!", message="Larga o celular 👀")
                        alert_sound.play()
                        last_alert = time.time()

        if time.time() < show_alert_until:
            cv2.putText(frame, "LARGA O CELULAR!", (30, 240),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)

        draw_button(frame)
        cv2.imshow("monitor", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    except cv2.error as e:
        print(f"Skipped a bad frame: {e}")
        continue

cap.release()
cv2.destroyAllWindows()
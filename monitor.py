import cv2
from ultralytics import YOLO
from plyer import notification
import time

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open camera. Check Windows camera privacy settings.")
    exit()

last_alert = 0

while True:
    ret, frame = cap.read()

    if not ret or frame is None or frame.size == 0:
        continue

    try:
        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = model.names[int(box.cls)]
                conf = float(box.conf)
                if cls == "cell phone" and conf > 0.5:
                    if time.time() - last_alert > 30:
                        notification.notify(title="Foco!", message="Larga o celular 👀")
                        last_alert = time.time()

        cv2.imshow("monitor", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    except cv2.error as e:
        print(f"Skipped a bad frame: {e}")
        continue

cap.release()
cv2.destroyAllWindows()
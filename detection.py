import cv2
from ultralytics import YOLO

# Model load karo (pehli baar run par khud download hoga, ~6 MB)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)   # video file ke liye: "videos/test.mp4"

if not cap.isOpened():
    print("Camera nahi khul raha!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection chalao
    results = model(frame, verbose=False)

    # Boxes aur labels ke saath annotated frame lo
    annotated = results[0].plot()

    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
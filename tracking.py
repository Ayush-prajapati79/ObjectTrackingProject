import cv2
import time
from ultralytics import YOLO

# Model load (CPU par yolov8n sabse fast hai)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)   # video file ke liye: "videos/test.mp4"

if not cap.isOpened():
    print("Camera nahi khul raha!")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tracking chalao (ByteTrack). persist=True se IDs frames ke beech yaad rehte hain
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",   # BoT-SORT ke liye: "botsort.yaml"
        imgsz=320,                  # CPU par speed ke liye
        conf=0.4,
        verbose=False
    )

    r = results[0]
    object_count = 0

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        # Kabhi kabhi ID pehli frame me None hoti hai, isliye check
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(boxes)

        for box, conf, cls, track_id in zip(boxes, confs, classes, ids):
            x1, y1, x2, y2 = box
            name = model.names[cls]
            id_text = f"ID: {track_id}" if track_id is not None else "ID: -"
            label = f"{name} | {id_text} | {conf*100:.0f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label ka background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        object_count = len(boxes)

    # FPS calculate
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Objects: {object_count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
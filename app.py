import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ---------- Source choose karo ----------
print("1 = Webcam")
print("2 = Video file")
choice = input("Source chuno (1/2): ").strip()

if choice == "2":
    path = input("Video ka path (jaise videos/test.mp4): ").strip().strip('"')
    cap = cv2.VideoCapture(path)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Source nahi khul raha!")
    exit()

# ---------- Model ----------
model = YOLO("yolov8n.pt")

# ---------- Output video setup ----------
os.makedirs("outputs", exist_ok=True)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS)
if src_fps is None or src_fps < 1:
    src_fps = 20   # webcam kabhi kabhi 0 batata hai

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")   # 24-hour format
output_path = f"outputs/{timestamp}.mp4"
writer = cv2.VideoWriter(output_path, fourcc, src_fps, (width, height))

prev_time = time.time()
fps_smooth = 0
unique_ids = set()   # ab tak dekhe gaye unique objects

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="my_tracker.yaml",
        imgsz=416,
        conf=0.4,
        verbose=False
    )
    r = results[0]
    object_count = 0

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(boxes)

        for box, conf, cls, tid in zip(boxes, confs, classes, ids):
            x1, y1, x2, y2 = box
            if tid is not None:
                unique_ids.add(int(tid))
            label = f"{model.names[cls]} | ID: {tid if tid is not None else '-'} | {conf*100:.0f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        object_count = len(boxes)

    # Smooth FPS
    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time
    fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps

    cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Objects now: {object_count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Total unique: {len(unique_ids)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

    writer.write(frame)
    cv2.imshow("Real-Time Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Output has been saved: {output_path}")
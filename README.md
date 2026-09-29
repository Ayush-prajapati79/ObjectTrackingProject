# Real-Time Object Detection and Tracking System



A real-time Computer Vision application that detects objects in a live webcam stream or video file, tracks them across frames using **unique IDs**, and displays the results with confidence scores, FPS and object counts.

---

## Demo

> Add a screenshot or GIF of your output here.
> Example: `![Demo](assets/demo.png)`

---

## Features

- Webcam or video-file input
- Object detection using **YOLOv8n** (80 COCO classes: person, car, bus, bottle, phone, etc.)
- Multi-object tracking with **ByteTrack** and persistent unique IDs
- On-screen bounding boxes, class label, track ID and confidence
- Live **FPS** (smoothed), **objects in frame** and **total unique objects**
- Annotated output video saved automatically with a timestamped name (`YYYYMMDD_HHMMSS.mp4`), so previous outputs are never overwritten
- Runs on a normal laptop CPU (no GPU required)

---

## How It Works

```
Camera / Video
      |
Read Frames (OpenCV)
      |
Object Detection (YOLOv8)  ->  Classes + Bounding Boxes + Confidence
      |
Object Tracking (ByteTrack) ->  Unique ID per object
      |
Draw Results (boxes, IDs, FPS, counts)
      |
Display + Save Output Video
```

- **Detection** answers: *what is present and where?*
- **Tracking** answers: *is this the same object as in the previous frames?*

---

## Tech Stack

| Component | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video capture, frame processing, display, video writing |
| Ultralytics YOLOv8 | Object detection |
| ByteTrack | Multi-object tracking |
| NumPy | Array and numerical operations |

---

## Project Structure

```
ObjectTrackingProject/
├── app.py              # Final application (source select, tracking, saving output)
├── webcam.py           # Stage 2: webcam capture test
├── detection.py        # Stage 3: YOLO object detection
├── tracking.py         # Stage 4-6: tracking, IDs, FPS, count
├── my_tracker.yaml     # Custom ByteTrack configuration
├── requirements.txt    # Python dependencies
├── outputs/            # Saved result videos (git-ignored)
└── videos/             # Input test videos (git-ignored)
```

`app.py` is the final version. The other scripts are the step-by-step learning stages of the project.

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/Ayush-prajapati79/ObjectTrackingProject.git
cd ObjectTrackingProject
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

The YOLO model (`yolov8n.pt`) downloads automatically on the first run.

---

## Usage

```bash
python app.py
```

You will be asked to choose the input source:

```
1 = Webcam
2 = Video file
```

- Choose `1` for the default webcam.
- Choose `2` and enter a path such as `videos/test.mp4`.
- Press **`q`** to stop. The output video is saved in `outputs/`.

### Output naming

Files are saved using a 24-hour timestamp:

```
outputs/20260929_153342.mp4
```

---

## Configuration

**Tracker settings** are in `my_tracker.yaml`. Important note: `track_buffer` is measured in **frames**, not seconds. At 30 FPS, `track_buffer: 60` keeps a lost object for about 2 seconds.

**Detection settings** in `app.py`:

| Parameter | Description |
|---|---|
| `conf` | Minimum confidence to accept a detection |
| `imgsz` | Inference image size. Smaller is faster, larger is more accurate |
| `classes` | Filter classes, e.g. `[2, 3, 5, 7]` for car, motorcycle, bus, truck |
| `tracker` | Tracker config file (`my_tracker.yaml`, `bytetrack.yaml`, `botsort.yaml`) |

---

## Performance

Tested on an Intel Core i5-1135G7 (CPU only, no dedicated GPU) with `yolov8n`, achieving roughly **20-30 FPS** on webcam input.

---

## Limitations

- ByteTrack uses motion and box overlap, not appearance. If an object leaves the frame or is fully hidden for longer than the track buffer, it may receive a **new ID** (ID switching).
- Objects that are very small, far away or poorly lit may be missed.
- Detection is limited to the 80 COCO classes of the pretrained model.

---

## Future Improvements

- Line-crossing counter for vehicles and people
- Zone-based alerts for restricted areas
- Export of tracking data (ID, class, time) to CSV
- Appearance-based Re-ID tracking (BoT-SORT) to reduce ID switching
- Web interface using Streamlit

---

## Applications

Traffic monitoring and vehicle counting, security surveillance, crowd and people counting, retail analytics, sports analysis and industrial safety monitoring.

---

## Author

**Ayush Prajapati**
GitHub: [@Ayush-prajapati79](https://github.com/Ayush-prajapati79)

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OpenCV](https://opencv.org/)

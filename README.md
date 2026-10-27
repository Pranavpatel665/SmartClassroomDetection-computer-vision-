# Smart Classroom Object Detection

A small computer-vision project that detects common classroom-relevant objects in an image, video, or webcam stream. It uses a pretrained Ultralytics YOLO model and OpenCV; it does not identify people.

## Features

- Detects people, backpacks, bottles, books, phones, chairs, clocks, keyboards, laptops, mice, remotes, and TVs.
- Saves annotated images or videos in `outputs/` and prints detection counts.
- Supports webcam input and optional live preview.
- Can be switched to all classes supported by the selected model.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The default YOLO weights (`yolo11n.pt`) are downloaded automatically the first time the program runs. An internet connection is needed for that initial download.

## Run

Place an image in `test_images/`, then run:

```powershell
python main.py --source test_images/classroom.jpg --show
```

Process a video:

```powershell
python main.py --source lecture.mp4 --output outputs/lecture_detected.mp4 --show
```

Use the default webcam (press `q` or `Esc` to stop):

```powershell
python main.py --source 0 --show
```

Omit `--show` to process and save without opening a preview. Use `--confidence 0.5` to change the confidence threshold, `--all-classes` to disable classroom filtering, or `--device cpu` to force CPU inference. Run `python main.py --help` to see all options.

## How It Works

1. OpenCV loads an image, video, or webcam frame.
2. YOLO predicts bounding boxes and confidence scores.
3. The default mode keeps the selected classroom-relevant COCO classes.
4. OpenCV/YOLO draws the detections, saves the result, and the program prints accumulated class counts.

## Limitations

The default model is trained on the general COCO dataset, not a classroom-specific dataset. It may not detect desks, whiteboards, or other classes absent from the model, and results vary with image quality, lighting, and occlusion. For a stronger classroom project, annotate classroom images and train a custom YOLO model, then pass its weights with `--model path/to/best.pt`. Treat detections as estimates, not as attendance or behavioral records.

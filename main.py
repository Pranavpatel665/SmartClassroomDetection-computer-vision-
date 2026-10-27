import cv2
from ultralytics import YOLO
from pathlib import Path

# Create folder for saving results
Path("results").mkdir(exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

print("\nSMART CLASSROOM OBJECT DETECTION")
print("1. Live Webcam")
print("2. Detect Objects in a Picture")

choice = input("Choose 1 or 2: ")

# LIVE WEBCAM
if choice == "1":
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Webcam not found!")
        raise SystemExit

    print("Webcam started. Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        result = model(frame, verbose=False)[0]
        output = result.plot()

        cv2.imshow("Smart Classroom Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

# PICTURE DETECTION
elif choice == "2":
    image_path = input("Enter picture path: ").strip().strip('"')

    image = cv2.imread(image_path)

    if image is None:
        print("Picture not found! Check the file path.")
        raise SystemExit

    result = model(image, verbose=False)[0]
    output = result.plot()

    print("\nDetected objects:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(f"{model.names[class_id]}: {confidence:.2%}")

    cv2.imwrite("results/detected_image.jpg", output)
    cv2.imshow("Picture Detection", output)
    print("Press any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Invalid choice. Please run again and enter 1 or 2.")
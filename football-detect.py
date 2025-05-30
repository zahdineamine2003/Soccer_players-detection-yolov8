import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8m-football.pt")

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Display results on the frame
    results.render()  # Draws bounding boxes and labels
    cv2.imshow("Football Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

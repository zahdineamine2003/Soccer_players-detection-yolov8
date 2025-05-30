import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8m-football.pt")  # Make sure this path is correct

# Load an image
img = cv2.imread("test.jpg")  # Replace 'test.jpg' with your image filename

# Perform inference
results = model(img)

# Visualize results
annotated_frame = results[0].plot()

# Show the output
cv2.imshow("YOLOv8 Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

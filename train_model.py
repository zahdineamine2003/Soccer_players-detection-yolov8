from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolov8m-football.pt")

# Train with further minimized parameters
model.train(
    data="data.yaml",     # Path to your dataset
    epochs=10,            # Fewer epochs to limit the total training time
    batch=2,              # Even smaller batch size
    imgsz=320,            # Further reduced image size
    device="cpu",         # Use CPU for training
    augment=True,         # Data augmentation to improve the model
    workers=1,            # Only 1 worker for loading data to reduce load
    lr0=0.005,            # Lower learning rate
    momentum=0.9,         # Standard momentum
    weight_decay=0.0005,  # Regularization
    save_period=5,        # Save the model periodically
    verbose=True          # Show detailed logs
)

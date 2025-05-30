import cv2
from ultralytics import YOLO
import torch
import pandas as pd
import time
import os

# Function to process the video and save detection stats
def process_video(video_path, model, output_csv, output_video=None):
    cap = cv2.VideoCapture(video_path)
    frame_skip = 2
    frame_count = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (640, 480))

    all_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (640, 480))

            start_pre = time.time()
            _ = frame.copy()  # simulate preprocessing
            end_pre = time.time()

            start_inf = time.time()
            results = model(frame, conf=0.4, iou=0.5)
            end_inf = time.time()

            start_post = time.time()
            annotated_frame = results[0].plot()
            end_post = time.time()

            # Count detected objects
            names = model.names
            detections = results[0].boxes.cls.tolist()
            counts = {name: 0 for name in names.values()}
            for det in detections:
                counts[names[int(det)]] += 1

            # Save frame data
            all_data.append({
                "frame": frame_count,
                "players": counts.get("player", 0),
                "goalkeepers": counts.get("goalkeeper", 0),
                "referees": counts.get("referee", 0),
                "balls": counts.get("ball", 0),
                "preprocess_time_ms": (end_pre - start_pre) * 1000,
                "inference_time_ms": (end_inf - start_inf) * 1000,
                "postprocess_time_ms": (end_post - start_post) * 1000,
                "total_time_ms": (end_post - start_pre) * 1000,
            })

            # Display
            cv2.imshow("YOLOv8 Video Detection", annotated_frame)

            if output_video:
                out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_video:
        out.release()
    cv2.destroyAllWindows()

    # Save CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved data to {output_csv}")

if __name__ == "__main__":
    video_path = 'video.mp4'
    output_video = 'output.mp4'
    output_csv = 'detection_stats.csv'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolov8m-football.pt")

    if device == "cuda":
        model.model.half()

    model.model.to(device)

    process_video(video_path, model, output_csv, output_video)

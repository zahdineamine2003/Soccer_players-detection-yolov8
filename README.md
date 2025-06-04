
# ⚽ Soccer Player Detection with YOLOv8

This project leverages **YOLOv8** to detect **players**, **referees**, **balls**, and **goalkeepers** in soccer match videos. It provides **real-time annotations**, detailed **performance statistics**, and exports the results as annotated videos and structured CSV files.

---

## 🖼️ Example Detections

<p align="center">
  <img src="https://raw.githubusercontent.com/zahdineamine2003/Soccer_players-detection-yolov8/d8f06dea2c7e18da8edfc32ec5cbf5ccc3020d46/detection.png" alt="Detection Sample" width="600"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/zahdineamine2003/Soccer_players-detection-yolov8/d8f06dea2c7e18da8edfc32ec5cbf5ccc3020d46/players_detection.png" alt="Player Detection Sample" width="600"/>
</p>

---

## 🎥 Output Demo (GIF Preview)

<p align="center">
  <img src="https://raw.githubusercontent.com/zahdineamine2003/Soccer_players-detection-yolov8/fca94f6ec0c9f103da317076c33f540b367d90bc/gif.gif" alt="Output GIF" width="600"/>
</p>

---

## 🚀 Features

✅ Detects the following soccer elements:
- Players  
- Goalkeepers  
- Referees  
- Balls  

✅ Includes:
- Real-time video annotation  
- Frame-by-frame statistics saved in CSV  
- Execution time for preprocessing, inference, and postprocessing  
- Optional annotated video export (`output.mp4`)  

---

## 📁 Project Structure

```

model/
├── football-detect.py        # Optional: Alternate detection script
├── video-test.py             # Main video detection script
├── video.mp4                 # Input video file
├── output.mp4                # Output with YOLOv8 annotations
├── gif.gif                   # GIF preview of the result
├── detection\_stats.csv       # Frame-by-frame detection stats
├── yolov8m-football.pt       # Custom trained YOLOv8 weights
├── detection.png             # Screenshot of detection
├── players\_detection.png     # Sample detection frame

````

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install ultralytics opencv-python pandas torch
````

---

## ▶️ How to Run

```bash
python model/video-test.py
```

---

## 🧠 Model Details

* **Model:** YOLOv8m (custom fine-tuned)
* **Confidence Threshold:** 0.4
* **IoU Threshold:** 0.5
* **Trained On:** Custom annotated soccer dataset

---

## 🖥️ Device Support

The script automatically selects the available device:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](model/LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome!
Feel free to submit:

* Pull requests for improvements
* Issues for bugs or feature suggestions
* Custom models for other sports





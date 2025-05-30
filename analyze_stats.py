import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

# --- Simulate synthetic detection data (~85-90% accurate) ---
data = []
num_frames = 50

for frame in range(num_frames):
    # Simulate 90–100% player detections
    detected_players = int(20 * random.uniform(0.9, 1.0))  # better detection
    # Goalkeeper detected 90% of the time
    detected_goalkeepers = 1 if random.random() < 0.9 else 0
    inference_time = random.uniform(10, 25)  # ms
    total_time = inference_time + random.uniform(5, 15)  # ms

    data.append({
        'frame': frame,
        'players': detected_players,
        'goalkeepers': detected_goalkeepers,
        'inference_time_ms': inference_time,
        'total_time_ms': total_time
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# --- Plot 1: Processing Time per Frame ---
plt.figure(figsize=(10,6))
plt.plot(df['frame'], df['inference_time_ms'], label='Inference Time (ms)')
plt.plot(df['frame'], df['total_time_ms'], label='Total Processing Time (ms)')
plt.xlabel('Frame')
plt.ylabel('Time (ms)')
plt.title('Processing Time per Frame')
plt.legend()
plt.grid()
plt.savefig('processing_times.png')
plt.show()

# --- Plot 2: Detected Players per Frame ---
plt.figure(figsize=(10,6))
plt.plot(df['frame'], df['players'], label='Detected Players')
plt.xlabel('Frame')
plt.ylabel('Number of Players')
plt.title('Detected Players per Frame')
plt.legend()
plt.grid()
plt.savefig('players_detection.png')
plt.show()

# --- Confusion Matrix Calculation ---
y_true = []
y_pred = []

for idx, row in df.iterrows():
    true_players = 20
    true_goalkeepers = 1

    pred_players = int(row['players'])
    pred_goalkeepers = int(row['goalkeepers'])

    y_true += ['player'] * true_players + ['goalkeeper'] * true_goalkeepers
    y_pred += ['player'] * pred_players + ['goalkeeper'] * pred_goalkeepers

    detected_total = pred_players + pred_goalkeepers
    true_total = true_players + true_goalkeepers
    if detected_total < true_total:
        y_pred += ['none'] * (true_total - detected_total)

# --- Confusion Matrix Display ---
labels = ['player', 'goalkeeper', 'none']
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix of Detection')
plt.savefig('confusion_matrix.png')
plt.show()

# --- Accuracy Calculation ---
correct_predictions = sum([cm[i][i] for i in range(len(labels))])
total_predictions = cm.sum()
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy of Detection: {accuracy:.2f}%")

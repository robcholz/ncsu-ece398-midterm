Then change the input shape from [200, 6] to [200, 3].

Compact updated spec:

Task:
N-event IMU event detector using accelerometer only
Input:
- Sampling rate: 100 Hz
- Window size: 2 seconds
- Input shape: [200, 3]
- Channels: ax, ay, az
- Stride: 0.25 sec
  Classes:
- N event classes
- idle / normal
- motion_artifact
- Total classes = N + 2
  Model:
  1D CNN
  Architecture:
  Input [200, 3]
  → Conv1D(32, kernel=5) + BatchNorm + ReLU + MaxPool1D(2)
  → Conv1D(64, kernel=5) + BatchNorm + ReLU + MaxPool1D(2)
  → Conv1D(128, kernel=3) + BatchNorm + ReLU
  → GlobalAveragePooling1D
  → Dense(64) + ReLU + Dropout(0.2)
  → Dense(N + 2) + Softmax
  Loss:
- Weighted cross entropy
  Optimizer:
- Adam, lr = 1e-3
  Inference:
- Sliding window every 0.1–0.25 sec
- Trigger if event probability > 0.7 for 2 consecutive windows
- Merge same-event detections within 0.8–1.0 sec
- Cooldown: 0.5 sec

Dataset target stays basically the same:

Minimum:
- 500 events per event class
- idle windows = 3–5× total event count
- motion artifact windows = 3–5× total event count
  Better:
- 1,000+ events per event class
- 10+ subjects
- split train/val/test by subject

With only ax, ay, az, add one optional derived channel later:

acc_mag = sqrt(ax² + ay² + az²)

Then input becomes:

[200, 4] = ax, ay, az, acc_mag

I would start with [200, 3], then compare against [200, 4].
Compact HAR Pipeline

Task

Input: chest IMU accelerometer
Signals: ax, ay, az
Model: 1D CNN
Goal: event classification
Classes: 7 events + background = 8 classes

⸻

1. Data

Raw sample:
[timestamp, ax, ay, az, label]

Windowing:

fs = 100 Hz
window = 2.0 s = 200 samples
stride = 0.25–0.5 s
baseline input = [200, 3]

Label rule:

if event covers >= 30–50% window → event label
else → background

⸻

2. Baseline

E0:
Input: ax, ay, az
Preprocessing: none
Model: Small 1D CNN

CNN:

Input [B, 3, 200]
Conv1D 3→16, k=5
BN + ReLU + MaxPool
Conv1D 16→32, k=5
BN + ReLU + MaxPool
Conv1D 32→64, k=3
BN + ReLU
GlobalAvgPool
Linear 64→8

Training:

loss = CrossEntropy
optimizer = Adam
lr = 1e-3
batch = 16/32
early stop by val macro-F1

⸻

3. Benchmark Metrics

Accuracy
Macro F1
Per-class F1
Per-class recall
Confusion matrix
Model size
Params
Inference latency
Peak RAM

Main metric:

Macro F1

⸻

4. Experiment Matrix

ID	Input	Preprocess	Model	Purpose
E0	ax, ay, az	none	Small CNN	baseline
E1	ax, ay, az, acc_mag	none	Small CNN	magnitude feature
E2	ax, ay, az	normalize	Small CNN	normalization
E3	ax, ay, az, acc_mag	normalize	Small CNN	best 4ch variant
E4	ax, ay, az	LPF 20Hz	Small CNN	filter test
E5	ax, ay, az, acc_mag	LPF 20Hz	Small CNN	filter + mag
E6	best input	best preprocess	Tiny CNN	edge candidate
E7	best input	best preprocess	Medium CNN	capacity test
E8	best setup	augmentation	Small CNN	robustness

⸻

5. Feature Variants

Baseline

X = [ax, ay, az]
shape = [200, 3]

Add magnitude

acc_mag = sqrt(ax² + ay² + az²)
X = [ax, ay, az, acc_mag]
shape = [200, 4]

⸻

6. Preprocessing Variants

Test in this order:

P0: none
P1: per-recording normalization
P2: low-pass filter 20Hz
P3: low-pass filter 15Hz
P4: median filter window=3/5

Avoid starting with heavy filtering.

⸻

7. CNN Variants

Tiny CNN

Conv 3/4→8
Conv 8→16
GAP
Linear

Small CNN

Conv 3/4→16
Conv 16→32
Conv 32→64
GAP
Linear

Medium CNN

Conv 3/4→32
Conv 32→64
Conv 64→128
GAP
Dropout
Linear

⸻

8. Augmentation

Use after baseline is stable:

Gaussian noise
random scaling 0.8–1.2
time shift ±100 ms
random crop/pad
axis dropout
small rotation / axis mixing

⸻

9. Evaluation Split

Do not only use random window split.

Use:

A: random window split
B: held-out recording
C: held-out person
D: held-out position

Most realistic:

held-out person / held-out position

⸻

10. Final Order

1. E0: raw ax/ay/az + Small CNN
2. Add acc_mag
3. Add normalization
4. Try filters
5. Try Tiny/Small/Medium CNN
6. Add augmentation
7. Benchmark on edge device

Main principle:

Change one factor per experiment.
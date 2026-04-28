# Tasks

## Model

Experiment Matrix: ID	Input	Preprocess	Model	Purpose

- [ ] E0	ax, ay, az	none	Small CNN	baseline
- [ ] E1	ax, ay, az, acc_mag	none	Small CNN	magnitude feature
- [ ] E2	ax, ay, az	normalize	Small CNN	normalization
- [ ] E3	ax, ay, az, acc_mag	normalize	Small CNN	best 4ch variant
- [ ] E4	ax, ay, az	LPF 20Hz	Small CNN	filter test
- [ ] E5	ax, ay, az, acc_mag	LPF 20Hz	Small CNN	filter + mag
- [ ] E6	best input	best preprocess	Tiny CNN	edge candidate
- [ ] E7	best input	best preprocess	Medium CNN	capacity test
- [ ] E8	best setup	augmentation	Small CNN	robustness

### Benchmark

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

## Deployment

- [ ] binary one: benchmark
- [ ] binary two: normal deployment

### Benchmark

#### Memory

- [ ] (william) flash size
- [ ] model binary size
- [ ] (william) peak heap usage (before_inference_free_heap-min_free_heap_during_inference)
- [ ] (william) Heap watermark: minimum free heap observed during inference
- [ ] (william) Stack watermark: minimum remaining stack for inference task

#### Latency

- [ ] (william) inference time
- [ ] (william) P95 inference time (derived from inference time)
- [ ] (william) cpu pressure=inference time/(1/inference freq)

#### Power

- ❌(impossible) use macos cli api to capture the usb current metrics

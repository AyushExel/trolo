---
sidebar_position: 1
---

# Inference

trolo provides both a CLI interface and Python API for inference.

## CLI Usage

Run inference on different inputs: 

### Basic inference with automatic model download

```bash
trolo predict --model dfine-n
```

### Inference on specific input

```bash
trolo predict --model dfine-n.pth --input img.jpg
```

### Support for multiple input types

```bash
trolo predict --model dfine-n.pth --input folder/ # image folder
trolo predict --model dfine-n.pth --input video.mp4 # video file
trolo predict --model dfine-n.pth --input 0 # webcam
```

## Smart Video Stream Inference

trolo implements streaming inference for videos to handle memory efficiently. This means you can process large videos without worrying about memory constraints.

## Python API

```python
from trolo.inference import DetectionPredictor

# Initialize predictor
predictor = DetectionPredictor(model="dfine-n")

# Get raw predictions
predictions = predictor.predict()

# Visualize results
plotted_preds = predictor.visualize(show=True, save=True)
```

### Prediction results

The `predict` method returns a list of dictionaries, where each dictionary contains the prediction results for an image.

```python
# Example prediction
preds = predictor.predict()

for pred in preds:
    boxes = pred['boxes']
    scores = pred['scores']
    labels = pred['labels']
    # ... #
```

### Useful model information like class names
Models trained with trolo are saved with useful information like class names. These can be accessed by:

```python
class_names = predictor.config.yaml_cfg['class_names']
```

You can then implement your own custom visualization function using these class names.
```python
# Example
predictor = DetectionPredictor(model="dfine-n")

preds = predictor.predict()
class_names = predictor.config.yaml_cfg['class_names']

for pred in preds:
    boxes = pred['boxes']
    scores = pred['scores']
    labels = pred['labels']
    # ... #
    for box, score, label in zip(boxes, scores, labels):
        print(f"Label: {class_names[label]}, Score: {score:.2f}")
```
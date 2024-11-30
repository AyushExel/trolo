---
sidebar_position: 1
---

# Quickstart

trolo is a framework for harnessing the power of transformers with YOLO models and other single-shot detectors!

## Installation

```bash
pip install trolo
```

## Key Features

- 🔥 Transformer-enhanced object detection
- 🎯 Single-shot detection capabilities  
- ⚡ High performance inference
- 🛠️ Easy to use CLI interface
- 🚀 Fast video stream inference
- 🧠 Automatic DDP handling

## Available Models

trolo provides several state-of-the-art detection models:

### 🔥 NEW 🔥 D-FINE
The D-FINE model redefines regression tasks in DETR-based detectors using Fine-grained Distribution Refinement (FDR).
[Official Paper](https://arxiv.org/abs/2410.13842) | [Official Implementation](https://github.com/Peterande/D-FINE)

![D-FINE Stats](https://raw.githubusercontent.com/Peterande/storage/master/figs/stats_padded.png)

| Model | AP<sup>val</sup> | Size | Latency |
|:---:|:---:|:---:|:---:|
| dfine-n | 42.8 | 4M | 2.12ms |
| dfine-s | 48.5 | 10M | 3.49ms |
| dfine-m | 52.3 | 19M | 5.62ms |

Find all the available models [here](/docs/docs/models/index.md).

# Inference

### CLI Interface

The basic command structure is:

```bash
trolo [command] [options]
```

For help:
```bash
trolo --help # general help
trolo [command] --help # command-specific help
```

### Python API

```python
from trolo.inference import DetectionPredictor

predictor = DetectionPredictor(model="dfine-n")
predictions = predictor.predict() # get predictions
plotted_preds = predictor.visualize(show=True, save=True) # visualize outputs
```

Visit the [inference](/features/inference) section for detailed usage instructions.


# Training

### CLI Interface

The basic command structure is:

```bash
trolo train [options]
```

Basic training examples:
```bash
trolo train --config dfine_n  # train using built-in config
trolo train --model dfine-n --dataset coco  # specify model and dataset separately
```

🔥 Automatic multi-GPU handling. Just specify the devices:
```bash
trolo train --device 0,1,2,3  # multi-GPU training
```

### Python API

```python
from trolo.trainers import DetectionTrainer

# Initialize trainer
trainer = DetectionTrainer(
    config="dfine_n"
)

# Start training
trainer.fit()  # single GPU
trainer.fit(device="0,1,2,3")  # multi-GPU
```

Visit the [Training](/features/training) section for detailed configuration options.


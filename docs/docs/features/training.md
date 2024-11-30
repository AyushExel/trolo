---
sidebar_position: 2
---

# Training

trolo provides flexible training options through both CLI and Python API interfaces.

## CLI Usage

Basic training command:
```bash
trolo train --config dfine_n
```

### Multi-GPU Training

trolo automatically handles distributed data parallel (DDP) training:
```bash
trolo train --config dfine_n --device 0,1,2,3
```

## Python API

```python
from trolo.trainers import DetectionTrainer

# Initialize trainer
trainer = DetectionTrainer(config="dfine_n")

# Start training
trainer.train()

# Multi-GPU training
trainer.train(device="0,1,2,3")
```

## Configuration

Training configurations can be customized through YAML files. See the configuration for details. 
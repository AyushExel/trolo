<img width="1024" alt="Screenshot 2024-11-20 at 2 38 59 AM" src="https://github.com/user-attachments/assets/73311b13-a624-4736-8472-b22318bcd6b0">

# trolo

A framework for harnessing the power of transformers with YOLO models and other single-shot detectors!

> **Note**: This project is currently in very early development and only supports the D-FINE model architecture.

## Installation

```bash
pip install trolo
```


## Features

- 🔥 Transformer-enhanced object detection
- 🎯 Single-shot detection capabilities  
- ⚡ High performance inference
- 🛠️ Easy to use CLI interface

## CLI Usage

The basic command structure is:

```bash
trolo [command] [options]
```


Example inference command:
```bash
trolo infer --model dfine_n_coco.pth --input ./data/images/ --output ./data/outputs/
trolo infer --model dfine_n_coco.pth --input video.mp4
```

Example training command:
```bash
trolo train --model-size large --dataset coco --use-automatic-mixed-precision --number-of-gpus 4
```

For help on available commands:
```bash
trolo --help
```


## Available Models

<details>
<summary><b>D-FINE</b></summary>

The D-FINE model redefines regression tasks in DETR-based detectors using Fine-grained Distribution Refinement (FDR).

Available configurations:
- D-FINE-N: Lightweight model
- D-FINE-S: Small model
- D-FINE-M: Medium model  
- D-FINE-L: Large model
- D-FINE-X: Extra large model

Reference configuration files:
```yaml:trolo/configs/yaml/dfine/dfine_hgnetv2_l_coco.yml
startLine: 1
endLine: 7
```

<summary><b>TROLO-24</b></summary>
**Currently under development.**


</details>

## Examples & Plots

Coming soon!

```
Performance comparison plot
```


```
Example detection outputs
```


## API Documentation

Coming soon! The API documentation will include:

- Model configuration
- Training pipelines
- Inference utilities
- Custom dataset integration

## Credits

This project builds upon several excellent open source projects:

- [D-FINE](https://github.com/Peterande/D-FINE): Original D-FINE model implementation
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR): Real-time DETR architecture
- [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection): Detection framework
- YOLO series models

## License

Apache 2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This is an early work in progress. Many features are still under development.

### Immidiate TODOs
- [ ] Add support for training with passing model and dataset configs separately
- [ ] 1-step multi-GPU training, by simply passing the number of GPUs to the CLI, and the trainer
- [ ] W&B logging
- [ ] Unit tests
- [ ] Docusaurus documentation

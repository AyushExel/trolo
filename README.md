<img width="512" alt="Screenshot 2024-11-20 at 2 38 59 AM" src="https://github.com/user-attachments/assets/73311b13-a624-4736-8472-b22318bcd6b0">

# trolo

A framework for harnessing the power of transformers with YOLO models and other single-shot detectors!

## Installation

```bash
pip install trolo
```

## Features

- 🔥 Transformer-enhanced object detection
- 🎯 Single-shot detection capabilities  
- ⚡ High performance inference
- 🛠️ Easy to use CLI interface
- 🚀 Fast video stream inference
- 🧠 Automatic DDP handling

## Quick Start

![pred_0](https://github.com/user-attachments/assets/144ae351-c520-4640-8081-3e9a0db9b432)

The CLI command structure is:

```bash
trolo [command] [options]
```
For detailed help:
```bash
trolo --help # for general help
trolo [command] --help # for command-specific help
```

### Inference

Example inference command:
```bash
trolo predict --model dfine-n # automatically downloads model from trolo model hub
```
Support for single image, image folder, and video input
```bash
trolo predict --model dfine-n.pth --input img.jpg # folder/ or video.mp4 or 0 (for webcam)
```

🔥 Smart Video stream inference - infers on videos in streaming mode so you never have to worry about memory issues!

Python API:
```python
from trolo.inference import DetectionPredictor

predictor = DetectionPredictor(model="dfine-n")
predictions = predictor.predict() # get predictions
poltted_preds = predictor.visualize(show=True, save=True) # or get visualized outputs
```
Visit Inference Docs for more details


### Training
<b>Example training command:</b>
```bash
trolo train --config dfine_n # automatically find the config file
```

🔥 Automatically handle DDP by simply passing the GPUs to the CLI
```bash
trolo train --config dfine_n --device 0,1,2,3 
```
That's it!

<b>Python API</b>

```python
from trolo.trainers import DetectionTrainer

trainer = DetectionTrainer(config="dfine_n") # or pass custom config path
trainer.train() # pass device = 0,1,2,3 to automatically handle DDP 
```

Visit Training Docs for more details


## Available Models

<details open>
<summary><b>D-FINE</b></summary>

The D-FINE model redefines regression tasks in DETR-based detectors using Fine-grained Distribution Refinement (FDR).
[Official Paper](https://arxiv.org/abs/2410.13842) | [Official Repo](https://github.com/Peterande/D-FINE)
![D-FINE Model Stats](https://raw.githubusercontent.com/Peterande/storage/master/figs/stats_padded.png)

( All models will be automatically downloaded when you pass the name for any task)
| Model | Dataset | AP<sup>val</sup> | #Params | Latency | GFLOPs |
| :---: | :---: | :---: |  :---: | :---: | :---: |
`dfine-n` | COCO | **42.8** | 4M | 2.12ms | 7
`dfine-s` | COCO | **48.5** | 10M | 3.49ms | 25
`dfine-m` | COCO | **52.3** | 19M | 5.62ms | 57
`dfine-l` | COCO | **54.0** | 31M | 8.07ms | 91
`dfine-x` | COCO | **55.8** | 62M | 12.89ms | 202

</details>

<details>
<summary><b>RT-DETR v3 (Coming Soon)</b></summary>
</details>

<details>
<summary><b>RT-DETR v2 (Coming Soon)</b></summary>
</details>

<details>
<summary><b>Trolo-2024 (WIP)</b></summary>
</details>


## Totally open source and free

TLDR: This is a non-profit project.Use it, modify it, copy it, do whatever you want with it. And if something doesn't allow you to do that, please open an issue.

<details>
<summary>More details</summary>
- Apache 2.0
- The license has simply been copied from official apache repo. Please open an issue if something doesn't allow you to use it. 
- This project is built on top of open licensed projects as mentioned below.

I intend to keep this project free and open source FOREVER. There are no plans for direct/indirect monetization of this project. I only accept sponsorships for compute resources to train models and perform independent research.
</details>

## Credits

This project builds upon several excellent open source projects:

- [D-FINE](https://github.com/Peterande/D-FINE): Original D-FINE model implementation
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR): Real-time DETR architecture
- [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection): Detection framework
</details>

<details>
<summary>More details</summary>
- The original trainer is based on D-fine with major modifications for handling pre-trained weights, DDP, and other features.
- The architecture is for D-fine is same as the original paper and repo. 
</details>


## Contributing

Contributions are most welcome! Please feel free to submit a Pull Request.

---

**Note**: This is an early work in progress. Many features are still under development.

### Immidiate TODOs
- [ ] Docusaurus documentation

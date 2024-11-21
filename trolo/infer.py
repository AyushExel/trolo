import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from trolo.loaders import YAMLConfig
from trolo.utils.smart_defaults import infer_model_config_path, infer_pretrained_model
from trolo.loaders.maps import MODEL_CONFIG_MAP, get_model_config_path

def draw_predictions(images, labels, boxes, scores, thrh=0.4):
    """Draw predictions on images"""
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill='blue',
            )

    return images


def process_image(model, input_path, device, format='torch', show=False, save=True):
    """Process a single image or folder of images
    
    Args:
        model: Model to use for inference
        input_path: Path to image file or folder containing images
        device: Device to run inference on
        format: Format of model ('torch', 'onnx', or 'trt')
        show: Whether to show the output image or video in a window
        save: Whether to save the output image or video
    """
    input_path = Path(input_path)
    
    # Get list of images to process
    images = []
    if input_path.is_file():
        # Single image
        try:
            im_pil = Image.open(input_path).convert('RGB')
            w, h = im_pil.size
            orig_size = torch.tensor([[w, h]]).to(device)
            images.append((input_path, im_pil, orig_size))
        except Exception as e:
            print(f"Could not process {input_path}: {e}")
            return
            
    elif input_path.is_dir():
        # Directory of images
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for file in input_path.iterdir():
            if file.is_file() and file.suffix.lower() in image_exts:
                try:
                    im_pil = Image.open(file).convert('RGB') 
                    w, h = im_pil.size
                    orig_size = torch.tensor([[w, h]]).to(device)
                    images.append((file, im_pil, orig_size))
                except Exception as e:
                    print(f"Could not process {file}: {e}")
                    continue
    else:
        print(f"Invalid input path: {input_path}")
        return

    if not images:
        print("No valid images found to process")
        return

    # Process all images
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    for file_path, im_pil, orig_size in images:
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        if format == 'torch':
            output = model(im_data, orig_size)
            result_images = draw_predictions([im_pil], output[0], output[1], output[2])
        else:
            blob = {
                'images': im_data,
                'orig_target_sizes': orig_size
            }
            output = model(blob)
            result_images = draw_predictions([im_pil], output['labels'], output['boxes'], output['scores'])

        if save:
            # Save result with original filename
            output_path = file_path.parent / f"{file_path.stem}_{format}_result{file_path.suffix}"
            result_images[0].save(output_path)
            print(f"Processed {file_path.name} -> {output_path.name}")

        if show:
            result_images[0].show()

def process_video(model, input_path, device, format='torch', show=False, save=True):
    """Process a video file"""
    cap = cv2.VideoCapture(input_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{format}_result.mp4', fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        if format == 'torch':
            output = model(im_data, orig_size)
            result_images = draw_predictions([frame_pil], output[0], output[1], output[2])
        else:
            blob = {
                'images': im_data,
                'orig_target_sizes': orig_size
            }
            output = model(blob)
            result_images = draw_predictions([frame_pil], output['labels'], output['boxes'], output['scores'])

        frame = cv2.cvtColor(np.array(result_images[0]), cv2.COLOR_RGB2BGR)
        
        if save:
            out.write(frame)
            
        if show:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    if save:
        out.release()
        print(f"Video processing complete. Result saved as '{format}_result.mp4'")
    if show:
        cv2.destroyAllWindows()

def load_model(model_path, format, device):
    """Load model based on format"""
    if format == 'torch':
        # Load model configuration
        model_name = Path(model_path).name
        cfg_path = get_model_config_path(model_name)
        cfg = YAMLConfig(cfg_path, resume=model_path)

        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # Load state into cfg.model first
        cfg.model.load_state_dict(state)

        # Create deployment model wrapper
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        # Initialize model and move to device
        model = Model().to(device)
        model.eval()
        
        return model
    elif format == 'onnx':
        return ort.InferenceSession(model_path)
    elif format == 'trt':
        # Reference TensorRT model loading implementation
        """
        Reference code block:
        ```python:D-FINE/tools/inference/trt_inf.py
        startLine: 40
        endLine: 54
        ```
        """
        pass
    else:
        raise ValueError(f"Unsupported format: {format}")

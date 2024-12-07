"""
This module contains utility functions for smart defaults.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Union, Optional, List
from trolo.loaders.maps import MODEL_CONFIG_MAP, get_model_config_path
from .assets import download_model


def find_config_files() -> Dict[str, Path]:
    """
    Find all config files in the package's config/ folder and installed locations.
    Returns a dict mapping config name to full path.
    """
    config_files = {}

    # Search package config directory
    pkg_root = Path(__file__).parent.parent
    config_dir = pkg_root / "configs"
    if config_dir.exists():
        for file in config_dir.rglob("*.yml"):
            config_files[file.name] = file

    return config_files


# Global map of available config files
CONFIG_FILES = find_config_files()

DEFAULT_DOWNLOAD_DIR = "."
DEFAULT_MODEL = "dfine_n.pth"
DEFAULT_OUTPUT_DIR = "output"

MODEL_HUB = "..."
SUPPORTED_IMG_FORMATS = [".jpg", ".jpeg", ".png", ".webp"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
SUPPORTED_EXPORT_TYPES = [".onnx"]

def infer_pretrained_model(model_path: str = DEFAULT_MODEL):
    """
    First check if the path exists. If so, use that otherwise download it from the model hub.
    """
    # First check if path exists directly
    if os.path.exists(model_path):
        return model_path

    # Check in default model download directory
    model_dir = Path(DEFAULT_DOWNLOAD_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    local_path = model_dir / model_path

    if local_path.exists():
        # Return the absolute path
        return str(local_path.resolve())
    # If model name is in hub models list, download it
    local_path = download_model(model_path)

    local_path = Path(local_path) if isinstance(local_path, str) else local_path
    if local_path and local_path.exists():
        return str(local_path.resolve())

    raise FileNotFoundError(
        f"Could not find model at {model_path} or in default model directory. "
        f"For pretrained models, please ensure the model name is found in the trolo model hub."
    )

def infer_input_path(input_path: str = None):
    """
    First check if the path exists, if not raise error.
    If path is not provided, use the data/samples/ directory form the installed package.
    """
    if input_path is None:
        pkg_root = Path(__file__).parent.parent
        input_path = pkg_root / "data" / "samples"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find input at {input_path}")

    return input_path

def infer_model_config_path(config_file: str = None):
    """
    Check if the config file exists in the package config directory. If not, search in the installed location
    configs/ directory recursively.
    """
    if config_file is None:
        return get_model_config_path(DEFAULT_MODEL)

    if os.path.exists(config_file) and config_file.endswith(".yml") or config_file.endswith(".yaml"):
        return config_file

    if config_file in MODEL_CONFIG_MAP:
        return get_model_config_path(config_file)

    raise FileNotFoundError(f"Could not find config file at {config_file} or in package config directory.")

def infer_device(device: Optional[str] = None):
    """
    If no device is provided, check if CUDA is available and use the first available GPU.
    Otherwise, use CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        else:
            return "cpu"

    return device

def infer_output_path(output_path: str = DEFAULT_OUTPUT_DIR):
    """
    Create output folder inside the given dir.
    Check if the output path exists, if not create it.
    if already exists, increment the output path by 1 until a free path is found.

    Example
    -------
    - input: ./outputs
    - output: ./outputs/output # in case outputs/ was empty
    - input: ./outputs/output_1 # if ./outputs/output already exists
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR

    # Convert to Path object for easier manipulation
    output_path = Path(output_path)

    # Create parent directory if it doesn't exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    # If base path doesn't exist, use it directly
    if not output_path.exists():
        output_path.mkdir(parents=True)
        return str(output_path)

    # Find next available numbered path
    counter = 1
    while True:
        new_path = output_path / f"output_{counter}"
        if not new_path.exists():
            new_path.mkdir(parents=True)
            return str(new_path)
        counter += 1

def infer_input_type(input_path: Union[str, Path]):
    """
    Infer the type of the input path.
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        return "folder"
    # Check if video
    elif input_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
        return "video"
    # Check if image
    elif input_path.suffix.lower() in SUPPORTED_IMG_FORMATS:
        return "image"
    # Add special case for webcam for upto 3 webcams
    elif input_path.name in ["0", "1", "2"]:
        return "webcam"
    else:
        raise ValueError(f"Unsupported input type: {input_path}")

def get_images_from_folder(input_path: str) -> List[str]:
    """
    Get all images from a folder non-recursively.
    """
    img_formats = []
    for ext in SUPPORTED_IMG_FORMATS:
        # Add case-insensitive matching by checking both upper and lowercase extensions
        img_formats.extend([str(p) for p in Path(input_path).glob(f"*{ext.lower()}")])
        img_formats.extend([str(p) for p in Path(input_path).glob(f"*{ext.upper()}")])
    return img_formats

def get_output_name(model_path:Union[str, Path] =  DEFAULT_MODEL, export_format : str = None, output_path : Optional[str] =  None ) -> str:

    model_path =  Path(infer_pretrained_model(model_path))

    export_dir  =  Path(output_path or  model_path.parent)
    if not export_dir.exists():
        export_dir.mkdir(parents=True)
    
    if not export_format:
        # LOGGER.warning(f"No export format specified. Exporting model as 'onnx by default.")
        export_format = model_path.suffix.lstrip('.')  # Use current suffix if none provided

    exported_model = export_dir / f"{model_path.stem}.{export_format}"

    # if exported_model.exists():
    #     raise  FileExistsError(
    #         f"The export file '{exported_model}' already exists."
    #     )

    return str(exported_model)

def format_size(size_input):
    if isinstance(size_input, int):
        size =  torch.tensor([[size_input, size_input]])
    elif isinstance(size_input, (list, tuple)) and len(size_input) == 2:
        size  = torch.tensor([size_input])
    elif isinstance(size_input, torch.Tensor) and size_input.dim() == 2:
        size  = size_input
    else:
        raise ValueError("Invalid size input. Must be an int, a tuple/list of two values, or a [N, 2] tensor.")
    return size

def format_input(data_input, batch_size = 1):
    if isinstance(data_input, int):  
        data = torch.rand(batch_size, 3, data_input, data_input) 
    elif isinstance(data_input, (list, tuple)) and len(data_input) == 2: 
        h, w = data_input
        data = torch.rand(batch_size, 3, h, w)  
    elif isinstance(data_input, torch.Tensor) and data_input.dim() == 4: 
        data = data_input
    else:
        raise ValueError("Invalid data input. Must be an int, a tuple/list of two values, or a 4D tensor.")
    return data

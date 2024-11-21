"""
This module contains utility functions for smart defaults.
"""

import os
import torch
from pathlib import Path
from typing import Dict
from trolo.loaders.maps import MODEL_CONFIG_MAP, get_model_config_path

def find_config_files() -> Dict[str, Path]:
    """
    Find all config files in the package's config/ folder and installed locations.
    Returns a dict mapping config name to full path.
    """
    config_files = {}
    
    # Search package config directory
    pkg_root = Path(__file__).parent.parent
    config_dir = pkg_root / 'configs'
    if config_dir.exists():
        for file in config_dir.rglob('*.yml'):
            config_files[file.name] = file
                
    return config_files

# Global map of available config files
CONFIG_FILES = find_config_files()

DEFAULT_DOWNLOAD_DIR = "~/.trolo/models"
DEFAULT_MODEL = "dfine_n_coco.pth"


MODEL_HUB = "..."
HUB_MODELS = ["fine_n_coco.pth"]



def infer_pretrained_model(model_path: str = DEFAULT_MODEL):
    """
    First check if the path exists. If so, use that otherwise download it from the model hub.
    """
    # First check if path exists directly
    if os.path.exists(model_path):
        return model_path
        
    # Check in default model download directory
    model_dir = Path.home() / '.trolo' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    local_path = model_dir / model_path
    
    if local_path.exists():
        return str(local_path)
        
    # If model name is in hub models list, download it
    if model_path in HUB_MODELS:
        # TODO: Implement actual download logic here
        # Download model from MODEL_HUB to local_path
        pass
        
        if local_path.exists():
            return str(local_path)
            
    raise FileNotFoundError(
        f"Could not find model at {model_path} or in default model directory. "
        f"For pretrained models, please ensure the model name is one of: {list(HUB_MODELS)}"
    )

def infer_input_path(input_path: str = None):
    """
    First check if the path exists, if not raise error. 
    If path is not provided, use the data/samples/ directory form the installed package.
    """
    if input_path is None:
        pkg_root = Path(__file__).parent.parent
        input_path = pkg_root / 'data' / 'samples'
    
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

    if os.path.exists(config_file) and config_file.endswith('.yml') or config_file.endswith('.yaml'):
        return config_file
    
    if config_file in MODEL_CONFIG_MAP:
        return get_model_config_path(config_file)
    
    raise FileNotFoundError(f"Could not find config file at {config_file} or in package config directory.")
    


def infer_device(device: str = None):
    """
    If no device is provided, check if CUDA is available and use the first available GPU.
    Otherwise, use CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            return f'cuda:{torch.cuda.current_device()}'
        else:
            return 'cpu'
            
    return device

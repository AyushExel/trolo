from pathlib import Path
from typing import Dict
from trolo.models.dfine.maps import MODEL_CONFIG_MAP as DFINE_MODEL_CONFIG_MAP

# Get package root directory
PKG_ROOT = Path(__file__).parent.parent
CONFIG_ROOT = PKG_ROOT / "configs" / "yaml"

# Map of model names to their config files
MODEL_CONFIG_MAP = {
    **DFINE_MODEL_CONFIG_MAP,
}

def get_model_config_path(model_path: str) -> str:
    """
    Get the absolute config file path for a given model name.
    Supports both short names (dfine-n) and full config names (dfine_hgnetv2_n_coco.yml)
    
    Args:
        model_name: Model name or config file name
        
    Returns:
        Absolute path to the config file
        
    Raises:
        ValueError: If model name is not found in mapping
    """
    # first check if the path exists
    if Path(model_path).exists() and model_path.endswith('.yml'):
        return str(model_path)

    model_name = Path(model_path).name.lower()
    
    # First try direct lookup
    if model_name in MODEL_CONFIG_MAP:
        return str(CONFIG_ROOT / MODEL_CONFIG_MAP[model_name])
    
    # Try with .yml extension
    if not model_name.endswith('.yml'):
        model_name += '.yml'
        if model_name in MODEL_CONFIG_MAP:
            return str(CONFIG_ROOT / MODEL_CONFIG_MAP[model_name])
            
    raise ValueError(f"Unknown model name: {model_name}. Available models: {list(MODEL_CONFIG_MAP.keys())}")


# Get package root directory
PKG_ROOT = Path(__file__).parent.parent
DATASET_CONFIG_ROOT = PKG_ROOT / "configs" / "yaml" / "dataset"

# Map of dataset names to their config files
DATASET_CONFIG_MAP = {
    # Base dataset names (short aliases)
    "coco": "coco_detection.yml",
    "ch": "crowdhuman_detection.yml",
    "obj365": "obj365_detection.yml",
    "custom": "custom_detection.yml",
    "dummy_coco": "dummy_coco.yml",
    
    # Full config file names also supported
    "coco_detection.yml": "coco_detection.yml",
    "crowdhuman_detection.yml": "crowdhuman_detection.yml",
    "obj365_detection.yml": "obj365_detection.yml",
    "custom_detection.yml": "custom_detection.yml",
    "dummy_coco.yml": "dummy_coco.yml",
}

def get_dataset_config_path(dataset_name: str) -> str:
    """
    Get the absolute config file path for a given dataset name.
    Supports both short names (coco) and full config names (coco_detection.yml)
    
    Args:
        dataset_name: Dataset name or config file name
        
    Returns:
        Absolute path to the dataset config file
        
    Raises:
        ValueError: If dataset name is not found in mapping
    """
    dataset_name = dataset_name.lower()
    
    # First try direct lookup
    if dataset_name in DATASET_CONFIG_MAP:
        return str(DATASET_CONFIG_ROOT / DATASET_CONFIG_MAP[dataset_name])
    
    # Try with .yml extension
    if not dataset_name.endswith('.yml'):
        dataset_name += '.yml'
        if dataset_name in DATASET_CONFIG_MAP:
            return str(DATASET_CONFIG_ROOT / DATASET_CONFIG_MAP[dataset_name])
            
    raise ValueError(f"Unknown dataset name: {dataset_name}. Available datasets: {list(DATASET_CONFIG_MAP.keys())}")
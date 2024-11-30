import os
import requests
from pathlib import Path
import logging
from typing import Optional

RELEASE_ASSETS_VER = "0.1.1"

def get_version() -> str:
    """Get package version from pyproject.toml"""
    try:
        from importlib.metadata import version
        return version("trolo")
    except ImportError:
        # Fallback for Python < 3.8
        import pkg_resources
        return pkg_resources.get_distribution("trolo").version


def download_model(model_name: str, output_dir: str = ".") -> Optional[str]:
    """
    Download a model from GitHub releases with flexible name matching.
    
    Args:
        model_name: Name of the model to download (e.g. 'dfine-n' or 'dfine_n')
        output_dir: Directory to save the downloaded model
        
    Returns:
        Path to downloaded model file if successful, None if failed
    """
    # Expand user directory if needed
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create both hyphen and underscore versions of the name
    name_variants = [
        model_name,
        model_name.replace('-', '_'),
        model_name.replace('_', '-')
    ]
    
    # Add .pth extension if not present
    name_variants = [
        n if n.endswith('.pth') else f"{n}.pth" 
        for n in set(name_variants)  # Remove duplicates
    ]
    
    # GitHub release base URL
    base_url = f"https://github.com/ayushexel/trolo/releases/download/{RELEASE_ASSETS_VER}"
    
    # Try each name variant
    for name in name_variants:
        # Check if file already exists locally
        local_path = Path(output_dir) / name
        if local_path.exists():
            logging.info(f"Model already exists at {local_path}")
            return str(local_path)
            
        # Try downloading from GitHub releases
        try:
            url = f"{base_url}/{name}"
            print(f"Downloading model from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save the file
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"Successfully downloaded model to {local_path}")
            return local_path
            
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            continue
            
    # If we get here, all download attempts failed
    print(f"Failed to download model {model_name}. "
                 f"Tried variants: {', '.join(name_variants)}")
    return None

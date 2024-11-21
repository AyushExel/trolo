import pytest
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from trolo.inference.detection import DetectionPredictor
from trolo.utils.smart_defaults import infer_input_path, infer_pretrained_model

DEFAULT_MODEL = "dfine_n_coco.pth"

@pytest.fixture(scope="session")
def model_path():
    """Get path to DFINE-N model weights"""
    try:
        return infer_pretrained_model(DEFAULT_MODEL)
    except FileNotFoundError:
        pytest.skip(f"Pretrained model {DEFAULT_MODEL} not found")

@pytest.fixture(scope="session")
def predictor(model_path):
    """Create predictor with DFINE-N model"""
    return DetectionPredictor(DEFAULT_MODEL)

@pytest.fixture
def sample_image():
    """Get sample image from package data"""
    sample_dir = infer_input_path()
    image_path = next(sample_dir.glob("*.jpg"))  # Get first jpg image
    if not image_path.exists():
        pytest.skip("No sample images found")
    return Image.open(image_path).convert('RGB')

def test_model_loading():
    """Test model loading and initialization"""
    predictor = DetectionPredictor(DEFAULT_MODEL)
    
    # Check model structure
    assert hasattr(predictor.model, 'backbone')
    
    # Check model is in eval mode
    assert not predictor.model.training
    
    # Check device placement
    assert next(predictor.model.parameters()).device == predictor.device

def test_predictor_preprocess(predictor, sample_image):
    """Test image preprocessing"""
    # Test single image
    output = predictor.preprocess(sample_image)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 1  # batch size
    assert output.shape[1] == 3  # channels
    assert all(s == 640 for s in output.shape[2:])  # HxW
    
    # Test normalization range
    assert output.min() >= -3 and output.max() <= 3  # Typical normalized image range
    
    # Test batch processing
    output_batch = predictor.preprocess([sample_image, sample_image])
    assert output_batch.shape[0] == 2
    assert output_batch.shape[1:] == output.shape[1:]

def test_predictor_inference(predictor, sample_image):
    """Test end-to-end inference"""
    with torch.no_grad():
        result = predictor.predict(sample_image)
    
    # Check output format
    assert isinstance(result, dict)
    assert all(k in result for k in ['boxes', 'scores', 'labels'])
    
    # Check output shapes
    batch_size = 1
    assert len(result['boxes'].shape) == 3  # (batch, num_boxes, 4)
    assert len(result['scores'].shape) == 2  # (batch, num_boxes)
    assert len(result['labels'].shape) == 2  # (batch, num_boxes)
    assert result['boxes'].shape[0] == batch_size
    
    # Check value ranges
    assert result['scores'].min() >= 0 and result['scores'].max() <= 1
    assert result['labels'].min() >= 0 and result['labels'].max() < 80  # COCO classes

def test_batch_inference(predictor, sample_image):
    """Test batch inference"""
    batch = [sample_image] * 2  # Create a batch of 2 identical images
    
    with torch.no_grad():
        result = predictor.predict(batch)
    
    assert result['boxes'].shape[0] == 2
    assert result['scores'].shape[0] == 2
    assert result['labels'].shape[0] == 2

@pytest.mark.parametrize("size", [(320, 320), (640, 480), (800, 600)])
def test_different_input_sizes(predictor, sample_image, size):
    """Test handling of different input sizes"""
    original_size = sample_image.size
    resized_image = sample_image.resize(size)
    
    with torch.no_grad():
        result = predictor.predict(resized_image)
    
    # Boxes should be scaled to input size (the resized dimensions)
    assert result['boxes'].max() <= max(size)

def test_invalid_inputs(predictor):
    """Test handling of invalid inputs"""
    with pytest.raises(Exception):
        predictor.predict(None)
    
    with pytest.raises(Exception):
        predictor.predict([])
    
    with pytest.raises(FileNotFoundError):
        predictor.predict("nonexistent.jpg")

@pytest.mark.parametrize("conf_threshold", [0.3, 0.5, 0.7])
def test_confidence_thresholding(predictor, sample_image, conf_threshold):
    """Test detection confidence thresholding"""
    with torch.no_grad():
        result = predictor.predict(sample_image, conf_threshold=conf_threshold)
    
    # All detection scores should be above threshold
    assert (result['scores'] >= conf_threshold).all()
#
#def test_model_download():
#    """Test model download functionality"""
#    # Remove model if exists
#    model_dir = Path.home() / '.trolo' / 'models'
#    model_path = model_dir / DEFAULT_MODEL
#    if model_path.exists():
#        model_path.unlink()
#    
#    # Should trigger download
#    predictor = DetectionPredictor(DEFAULT_MODEL)
#    assert model_path.exists()
#    
#    # Test model works
#    sample_dir = infer_input_path()
#    image_path = next(sample_dir.glob("*.jpg"))
#    if image_path.exists():
#        result = predictor.predict(Image.open(image_path).convert('RGB'))
#        assert isinstance(result, dict)
#        assert all(k in result for k in ['boxes', 'scores', 'labels'])

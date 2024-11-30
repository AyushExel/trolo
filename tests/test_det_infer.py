import pytest
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import cv2

from trolo.inference.detection import DetectionPredictor
from trolo.inference.video import VideoStream
from trolo.utils.smart_defaults import infer_input_path, infer_pretrained_model

DEFAULT_MODEL = "dfine_n.pth"

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

#def test_model_loading():
#    """Test model loading and initialization"""
#    predictor = DetectionPredictor(DEFAULT_MODEL)
#    
#    # Check model structure
#    assert hasattr(predictor.model, 'backbone')
#    
#    # Check model is in eval mode
#    assert not predictor.model.training
#    
#    # Check device placement
#    assert next(predictor.model.parameters()).device == predictor.device
#
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
    assert isinstance(result, list)
    assert len(result) == 1  # Single image input = single result
    result = result[0]  # Get first prediction
    
    assert isinstance(result, dict)
    assert all(k in result for k in ['boxes', 'scores', 'labels'])
    
    # Check output shapes
    assert len(result['boxes'].shape) == 2  # (num_boxes, 4)
    assert len(result['scores'].shape) == 1  # (num_boxes,)
    assert len(result['labels'].shape) == 1  # (num_boxes,)
    
    # Check value ranges
    assert result['scores'].min() >= 0 and result['scores'].max() <= 1
    assert result['labels'].min() >= 0 and result['labels'].max() < 80  # COCO classes

def test_batch_inference(predictor, sample_image):
    """Test batch inference"""
    batch = [sample_image] * 2  # Create a batch of 2 identical images
    
    with torch.no_grad():
        result = predictor.predict(batch)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(pred, dict) for pred in result)
    assert all(pred['boxes'].shape[0] == result[0]['boxes'].shape[0] for pred in result)

@pytest.mark.parametrize("size", [(320, 320), (640, 480), (800, 600)])
def test_different_input_sizes(predictor, sample_image, size):
    """Test handling of different input sizes"""
    original_size = sample_image.size
    resized_image = sample_image.resize(size)
    
    with torch.no_grad():
        result = predictor.predict(resized_image)
    
    # Get first prediction
    result = result[0]
    
    # Boxes should be scaled to input size (the resized dimensions)
    assert result['boxes'][:, [0, 2]].max() <= size[0]  # x coordinates
    assert result['boxes'][:, [1, 3]].max() <= size[1]  # y coordinates


@pytest.mark.parametrize("conf_threshold", [0.3, 0.5, 0.7])
def test_confidence_thresholding(predictor, sample_image, conf_threshold):
    """Test detection confidence thresholding"""
    with torch.no_grad():
        result = predictor.predict(sample_image, conf_threshold=conf_threshold)
    
    # Get first prediction
    result = result[0]
    
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

@pytest.fixture
def predictor():
    return DetectionPredictor(
        model="models/dfine_n.pth",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

@pytest.fixture
def sample_video(tmp_path):
    """Create a sample video file for testing"""
    video_path = tmp_path / "test.mp4"
    frames = 30
    height, width = 480, 640
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    
    for _ in range(frames):
        # Create random frame with a rectangle
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), 2)
        out.write(frame)
    
    out.release()
    return video_path


def test_video_prediction_streaming(predictor, sample_video):
    """Test video prediction in streaming mode"""
    frame_count = 0
    batch_sizes = [1, 4, 8]
    
    for batch_size in batch_sizes:
        frame_count = 0
        for predictions, frames in predictor.predict(
            str(sample_video),
            batch_size=batch_size,
            stream=True,
            return_inputs=True
        ):
            # Check batch dimensions
            assert len(predictions) == len(frames) <= batch_size
            
            # Check prediction format
            for pred in predictions:
                assert all(k in pred for k in ['boxes', 'scores', 'labels'])
                assert all(isinstance(v, torch.Tensor) for v in pred.values())
            
            # Check frame format
            assert all(isinstance(f, Image.Image) for f in frames)
            frame_count += len(frames)
    
        # Verify we processed all frames
        cap = cv2.VideoCapture(str(sample_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == total_frames



def test_error_handling(predictor):
    """Test error cases"""
    with pytest.raises(ValueError):
        # Test invalid input type
        predictor.predict("nonexistent.xyz")
    
    with pytest.raises(ValueError):
        # Test stream=True with image input
        predictor.predict("image.jpg", stream=True)


import pytest
from pathlib import Path
import torch

from trolo.utils.dummy_data import create_dummy_coco_dataset
from trolo.trainers.detection import DetectionTrainer
from trolo.loaders import YAMLConfig
from trolo.loaders.maps import get_dataset_config_path

@pytest.fixture(scope="session")
def dummy_dataset():
    """Create a temporary dummy COCO dataset"""
    dataset_path = create_dummy_coco_dataset(
        root_dir="./data/dummy_coco",
        num_images=10,
        num_objects_per_image=3
    )
    return dataset_path

def test_training_loop(dummy_dataset):
    """Test basic training functionality using combined config"""
    # Initialize trainer with combined config
    trainer = DetectionTrainer(
        config="dfine_n_coco_dummy"  # Using the combined dummy config
    )
    
    # Run training
    trainer.fit()
    
    # Basic checks
    assert trainer.last_epoch >= 0  # Should have completed at least one epoch
    
    
    # Check that evaluation metrics were computed
    assert hasattr(trainer.evaluator, 'coco_eval')
    assert 'bbox' in trainer.evaluator.coco_eval
    
    # Check that output files were created
    output_dir = Path(trainer.cfg.output_dir)
    assert (output_dir / "log.txt").exists()
    assert (output_dir / "eval").exists()

#def test_training_separate_configs(dummy_dataset):
#    """Test training with separate model and dataset configs"""
#    # Initialize trainer with separate model and dataset configs
#    trainer = DetectionTrainer(
#        model="dfine-n",      # Using model name from MODEL_CONFIG_MAP
#        dataset="dummy_coco"  # Using dataset name
#    )
#    
#    # Run training
#    trainer.fit()
#    
#    # Basic checks
#    assert trainer.epoch >= 0
#    assert hasattr(trainer.evaluator, 'coco_eval')
#    assert 'bbox' in trainer.evaluator.coco_eval

def test_validation_only(dummy_dataset):
    """Test validation-only functionality"""
    trainer = DetectionTrainer(
        config="dfine_n_coco_dummy"
    )
    
    # Run validation only
    trainer.val()
    
    # Check that evaluation metrics were computed
    assert hasattr(trainer.evaluator, 'coco_eval')
    assert 'bbox' in trainer.evaluator.coco_eval
    
    # Check evaluation output was saved
    output_dir = Path(trainer.cfg.output_dir)
    assert (output_dir / "eval.pth").exists() 
import pytest
from pathlib import Path
from pycocotools.coco import COCO
from trolo.utils.dummy_data import create_dummy_coco_dataset

@pytest.fixture(scope="session")
def dummy_dataset():
    return create_dummy_coco_dataset()

def test_dataset_structure(dummy_dataset):
    # Check directory structure
    assert (dummy_dataset / "train2017").exists()
    assert (dummy_dataset / "val2017").exists()
    assert (dummy_dataset / "annotations").exists()

def test_annotations(dummy_dataset):
    # Test train annotations
    train_ann = COCO(str(dummy_dataset / "annotations/instances_train2017.json"))
    assert len(train_ann.getImgIds()) > 0
    assert len(train_ann.getCatIds()) == 80
    
    # Test val annotations
    val_ann = COCO(str(dummy_dataset / "annotations/instances_val2017.json"))
    assert len(val_ann.getImgIds()) > 0
    assert len(val_ann.getCatIds()) == 80

def test_images(dummy_dataset):
    # Check if images exist and are valid
    train_ann = COCO(str(dummy_dataset / "annotations/instances_train2017.json"))
    for img_id in train_ann.getImgIds():
        img_info = train_ann.loadImgs(img_id)[0]
        img_path = dummy_dataset / "train2017" / img_info["file_name"]
        assert img_path.exists() 
import json
import os
from PIL import Image
import numpy as np
from pathlib import Path

def create_dummy_coco_dataset(root_dir="./data/dummy_coco", num_images=5, num_objects_per_image=3):
    """Create a minimal COCO format dataset for testing"""
    
    # Create directory structure
    root_dir = Path(root_dir)
    (root_dir / "train2017").mkdir(parents=True, exist_ok=True)
    (root_dir / "val2017").mkdir(parents=True, exist_ok=True)
    (root_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # Define categories (using a subset of COCO categories)
    categories = [
        {"id": 0, "name": "person", "supercategory": "person"},
        {"id": 1, "name": "car", "supercategory": "vehicle"},
        {"id": 2, "name": "dog", "supercategory": "animal"}
    ]

    def create_split(split_name):
        images = []
        annotations = []
        ann_id = 1

        # Create dummy images and annotations
        for img_id in range(1, num_images + 1):
            # Create a random color image
            img_size = (640, 480)
            img = Image.fromarray(np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8))
            img_path = root_dir / f"{split_name}2017" / f"{img_id:012d}.jpg"
            img.save(img_path)

            images.append({
                "id": img_id,
                "file_name": f"{img_id:012d}.jpg",
                "height": img_size[1],
                "width": img_size[0]
            })

            # Create random annotations for this image
            for _ in range(num_objects_per_image):
                # Random box dimensions
                x = np.random.randint(0, img_size[0] - 100)
                y = np.random.randint(0, img_size[1] - 100)
                w = np.random.randint(50, 100)
                h = np.random.randint(50, 100)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": np.random.randint(0, len(categories)),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

        # Create annotation file
        ann_file = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        with open(root_dir / "annotations" / f"instances_{split_name}2017.json", "w") as f:
            json.dump(ann_file, f)

    # Create train and val splits
    create_split("train")
    create_split("val")

    # Create a minimal config file
    config = {
        "dataset": {
            "train": {
                "img_folder": str(root_dir / "train2017"),
                "ann_file": str(root_dir / "annotations/instances_train2017.json")
            },
            "val": {
                "img_folder": str(root_dir / "val2017"),
                "ann_file": str(root_dir / "annotations/instances_val2017.json")
            }
        }
    }

    with open(root_dir / "config.yml", "w") as f:
        import yaml
        yaml.dump(config, f)

    return root_dir

if __name__ == "__main__":
    dataset_path = create_dummy_coco_dataset()
    print(f"Created dummy COCO dataset at {dataset_path}")

import json
import os
from PIL import Image
import numpy as np
from pathlib import Path

def create_dummy_coco_dataset(root_dir="./data/dummy_coco", num_images=5, num_objects_per_image=3):
    """Create a COCO format dataset for testing that matches COCO category IDs"""
    
    # Create directory structure
    root_dir = Path(root_dir)
    (root_dir / "train2017").mkdir(parents=True, exist_ok=True)
    (root_dir / "val2017").mkdir(parents=True, exist_ok=True)
    (root_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # Define all 80 COCO categories with proper IDs
    categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
        {"id": 3, "name": "car", "supercategory": "vehicle"},
        {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
        {"id": 5, "name": "airplane", "supercategory": "vehicle"},
        {"id": 6, "name": "bus", "supercategory": "vehicle"},
        {"id": 7, "name": "train", "supercategory": "vehicle"},
        {"id": 8, "name": "truck", "supercategory": "vehicle"},
        {"id": 9, "name": "boat", "supercategory": "vehicle"},
        {"id": 10, "name": "traffic light", "supercategory": "outdoor"},
        {"id": 11, "name": "fire hydrant", "supercategory": "outdoor"},
        {"id": 13, "name": "stop sign", "supercategory": "outdoor"},
        {"id": 14, "name": "parking meter", "supercategory": "outdoor"},
        {"id": 15, "name": "bench", "supercategory": "outdoor"},
        {"id": 16, "name": "bird", "supercategory": "animal"},
        {"id": 17, "name": "cat", "supercategory": "animal"},
        {"id": 18, "name": "dog", "supercategory": "animal"},
        {"id": 19, "name": "horse", "supercategory": "animal"},
        {"id": 20, "name": "sheep", "supercategory": "animal"},
        {"id": 21, "name": "cow", "supercategory": "animal"},
        {"id": 22, "name": "elephant", "supercategory": "animal"},
        {"id": 23, "name": "bear", "supercategory": "animal"},
        {"id": 24, "name": "zebra", "supercategory": "animal"},
        {"id": 25, "name": "giraffe", "supercategory": "animal"},
        {"id": 27, "name": "backpack", "supercategory": "accessory"},
        {"id": 28, "name": "umbrella", "supercategory": "accessory"},
        {"id": 31, "name": "handbag", "supercategory": "accessory"},
        {"id": 32, "name": "tie", "supercategory": "accessory"},
        {"id": 33, "name": "suitcase", "supercategory": "accessory"},
        {"id": 34, "name": "frisbee", "supercategory": "sports"},
        {"id": 35, "name": "skis", "supercategory": "sports"},
        {"id": 36, "name": "snowboard", "supercategory": "sports"},
        {"id": 37, "name": "sports ball", "supercategory": "sports"},
        {"id": 38, "name": "kite", "supercategory": "sports"},
        {"id": 39, "name": "baseball bat", "supercategory": "sports"},
        {"id": 40, "name": "baseball glove", "supercategory": "sports"},
        {"id": 41, "name": "skateboard", "supercategory": "sports"},
        {"id": 42, "name": "surfboard", "supercategory": "sports"},
        {"id": 43, "name": "tennis racket", "supercategory": "sports"},
        {"id": 44, "name": "bottle", "supercategory": "kitchen"},
        {"id": 46, "name": "wine glass", "supercategory": "kitchen"},
        {"id": 47, "name": "cup", "supercategory": "kitchen"},
        {"id": 48, "name": "fork", "supercategory": "kitchen"},
        {"id": 49, "name": "knife", "supercategory": "kitchen"},
        {"id": 50, "name": "spoon", "supercategory": "kitchen"},
        {"id": 51, "name": "bowl", "supercategory": "kitchen"},
        {"id": 52, "name": "banana", "supercategory": "food"},
        {"id": 53, "name": "apple", "supercategory": "food"},
        {"id": 54, "name": "sandwich", "supercategory": "food"},
        {"id": 55, "name": "orange", "supercategory": "food"},
        {"id": 56, "name": "broccoli", "supercategory": "food"},
        {"id": 57, "name": "carrot", "supercategory": "food"},
        {"id": 58, "name": "hot dog", "supercategory": "food"},
        {"id": 59, "name": "pizza", "supercategory": "food"},
        {"id": 60, "name": "donut", "supercategory": "food"},
        {"id": 61, "name": "cake", "supercategory": "food"},
        {"id": 62, "name": "chair", "supercategory": "furniture"},
        {"id": 63, "name": "couch", "supercategory": "furniture"},
        {"id": 64, "name": "potted plant", "supercategory": "furniture"},
        {"id": 65, "name": "bed", "supercategory": "furniture"},
        {"id": 67, "name": "dining table", "supercategory": "furniture"},
        {"id": 70, "name": "toilet", "supercategory": "furniture"},
        {"id": 72, "name": "tv", "supercategory": "electronic"},
        {"id": 73, "name": "laptop", "supercategory": "electronic"},
        {"id": 74, "name": "mouse", "supercategory": "electronic"},
        {"id": 75, "name": "remote", "supercategory": "electronic"},
        {"id": 76, "name": "keyboard", "supercategory": "electronic"},
        {"id": 77, "name": "cell phone", "supercategory": "electronic"},
        {"id": 78, "name": "microwave", "supercategory": "appliance"},
        {"id": 79, "name": "oven", "supercategory": "appliance"},
        {"id": 80, "name": "toaster", "supercategory": "appliance"},
        {"id": 81, "name": "sink", "supercategory": "appliance"},
        {"id": 82, "name": "refrigerator", "supercategory": "appliance"},
        {"id": 84, "name": "book", "supercategory": "indoor"},
        {"id": 85, "name": "clock", "supercategory": "indoor"},
        {"id": 86, "name": "vase", "supercategory": "indoor"},
        {"id": 87, "name": "scissors", "supercategory": "indoor"},
        {"id": 88, "name": "teddy bear", "supercategory": "indoor"},
        {"id": 89, "name": "hair drier", "supercategory": "indoor"},
        {"id": 90, "name": "toothbrush", "supercategory": "indoor"}
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
                # Get a random category
                category = categories[np.random.randint(0, len(categories))]
                
                # Random box dimensions
                x = np.random.randint(0, img_size[0] - 100)
                y = np.random.randint(0, img_size[1] - 100)
                w = np.random.randint(50, 100)
                h = np.random.randint(50, 100)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category["id"],  # Using actual COCO category ID
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

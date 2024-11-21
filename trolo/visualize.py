from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Dict, List, Union

def visualize_detections(
    image: Union[Image.Image, List[Image.Image]], 
    predictions: Dict[str, "torch.Tensor"], 
    confidence_threshold: float = 0.5,
    colors: Dict[int, tuple] = None,
    class_names: Dict[int, str] = None
) -> Union[Image.Image, List[Image.Image]]:
    """
    Visualize detection results on image(s)
    
    Args:
        image: PIL Image or list of PIL Images
        predictions: Dictionary containing 'boxes', 'scores', 'labels'
        confidence_threshold: Minimum confidence score to display
        colors: Optional dictionary mapping class ids to RGB colors
        class_names: Optional dictionary mapping class ids to class names
    
    Returns:
        PIL Image(s) with drawn detections
        
    Examples:
        # Single image inference and visualization
        >>> from PIL import Image
        >>> from trolo.inference.detection import DetectionPredictor
        >>> 
        >>> # Load model and image
        >>> predictor = DetectionPredictor("dfine_n_coco.pth")
        >>> image = Image.open("path/to/image.jpg").convert('RGB')
        >>> 
        >>> # Run inference
        >>> predictions = predictor.predict(image)
        >>> 
        >>> # Basic visualization
        >>> result = visualize_detections(image, predictions)
        >>> result.show()
        >>> result.save("output.jpg")
        
        # Batch processing with custom classes and colors
        >>> # Define COCO class names
        >>> class_names = {
        ...     0: 'person',
        ...     1: 'bicycle',
        ...     2: 'car',
        ...     # ... add more classes as needed
        >>> }
        >>> 
        >>> # Define custom colors (RGB)
        >>> colors = {
        ...     0: (255, 0, 0),    # Red for person
        ...     1: (0, 255, 0),    # Green for bicycle
        ...     2: (0, 0, 255),    # Blue for car
        ...     # ... add more colors as needed
        >>> }
        >>> 
        >>> # Process multiple images
        >>> images = [Image.open(f) for f in ["img1.jpg", "img2.jpg"]]
        >>> predictions = predictor.predict(images)
        >>> 
        >>> # Visualize with custom settings
        >>> results = visualize_detections(
        ...     images, 
        ...     predictions,
        ...     confidence_threshold=0.7,
        ...     colors=colors,
        ...     class_names=class_names
        >>> )
        >>> 
        >>> # Save results
        >>> for i, result in enumerate(results):
        ...     result.save(f"output_{i}.jpg")
    """
    if isinstance(image, Image.Image):
        image = [image]
        for k in predictions:
            predictions[k] = predictions[k].unsqueeze(0)
            
    # Default color palette
    if colors is None:
        colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(80)}
    
    # Try to load a larger font, fallback to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 24)  # Increased font size
    except:
        font = ImageFont.load_default()
        
    result_images = []
    
    for idx, img in enumerate(image):
        # Create copy of image for drawing
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        
        # Get predictions for this image
        boxes = predictions['boxes'][idx]
        scores = predictions['scores'][idx]
        labels = predictions['labels'][idx]
        
        # Filter by confidence
        mask = scores >= confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Draw each detection
        for box, score, label in zip(boxes, scores, labels):
            # Convert from cxcywh to xyxy format
            cx, cy, w, h = box.tolist()
            x0 = cx - w/2
            y0 = cy - h/2
            x1 = cx + w/2
            y1 = cy + h/2
            
            # Convert to integers and ensure within image bounds
            x0 = max(0, min(int(x0), img.width))
            y0 = max(0, min(int(y0), img.height))
            x1 = max(0, min(int(x1), img.width))
            y1 = max(0, min(int(y1), img.height))
            
            # Skip invalid boxes
            if x1 <= x0 or y1 <= y0:
                continue
                
            # Get color for current label
            color = colors[int(label.item())]
            
            # Draw box with thicker width
            draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
            
            # Prepare label text
            label_text = f"{class_names[int(label.item())] if class_names else label.item()}"
            conf_text = f"{score.item():.2f}"
            
            # Get text sizes
            label_bbox = draw.textbbox((0, 0), label_text, font=font)
            conf_bbox = draw.textbbox((0, 0), conf_text, font=font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]
            conf_width = conf_bbox[2] - conf_bbox[0]
            
            # Draw label background
            label_bg_color = tuple(int(c * 0.7) for c in color)  # Darker version of box color
            draw.rectangle(
                [x0, max(0, y0 - label_height - 4), x0 + label_width + 4, y0],
                fill=label_bg_color
            )
            
            # Draw confidence background
            draw.rectangle(
                [x0 + label_width + 4, max(0, y0 - label_height - 4), 
                 x0 + label_width + conf_width + 8, y0],
                fill=(50, 50, 50)  # Dark gray
            )
            
            # Draw text with a small offset from the top of the box
            draw.text(
                (x0 + 2, max(0, y0 - label_height - 2)), 
                label_text,
                fill=(255, 255, 255),  # White text
                font=font
            )
            draw.text(
                (x0 + label_width + 6, max(0, y0 - label_height - 2)),
                conf_text,
                fill=(255, 255, 255),  # White text
                font=font
            )
            
            #print(f"Original box (cxcywh): {box.tolist()}")
            #print(f"Converted box (xyxy): [{x0}, {y0}, {x1}, {y1}]")
            #print(f"Image size: {img.size}")
            
        result_images.append(draw_img)
        
    return result_images[0] if len(result_images) == 1 else result_images

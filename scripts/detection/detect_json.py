from ultralytics import YOLO
import torch
import cv2
from pathlib import Path
import json

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 0
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("No GPU detected, using CPU")

    # Load YOLO model
    print("\nLoading YOLO model...")
    
    # Load the trained model from train.py
    model_path = Path('runs/detect/training/weights/best.pt')
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}\n"
            "Run train.py first to generate the model."
        )
    
    print(f"Using trained model: {model_path}")
    model = YOLO(str(model_path))
    
    # Get class names from the model
    class_names = model.names
    
    # Source images - use proper test set
    test_dirs = [
        Path('./data/training/datasets/aircraft-components/images/test'),  # Proper test set
        Path('./data/testing/images'),  # Legacy location
    ]
    source_dir = next((d for d in test_dirs if d.exists() and list(d.glob('*.png'))), None)
    if source_dir is None:
        raise FileNotFoundError(
            "No test images found. Check paths.\n"
            + "Checked: " + ", ".join(str(d) for d in test_dirs)
        )
    
    print(f"Using test images from: {source_dir}")
    output_dir = Path('demo/all_detections')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(source_dir.glob('*.png')) + list(source_dir.glob('*.jpg'))
    
    print(f"\n{'='*80}")
    print(f"Processing {len(image_files)} images with YOLO detection")
    print(f"{'='*80}\n")
    
    all_detections = []
    
    for img_idx, img_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"Image {img_idx}/{len(image_files)}: {img_path.name}")
        print(f"{'='*80}")
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Run YOLO detection
        results = model.predict(source=str(img_path), conf=0.3, device=device, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            print("No components detected")
            continue
        
        result = results[0]
        boxes = result.boxes
        
        print(f"Found {len(boxes)} components")
        
        image_detections = []
        
        # Process each detected component
        for box_idx, box in enumerate(boxes, 1):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            
            print(f"  Component {box_idx}: {class_name} (conf: {conf:.2%})")
            print(f"    Bbox: xmin={x1}, ymin={y1}, xmax={x2}, ymax={y2}")
            
            # Draw bounding box with smaller thickness (1 instead of 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Add text label with smaller font (0.3 instead of 0.5)
            label = f"{class_name} {conf:.2%}"
            
            # Create text background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            # Add to detections list
            image_detections.append({
                'unique_id': f"{img_path.stem}_component_{box_idx}",
                'label': class_name,
                'bbox': {
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2
                },
                'confidence': conf
            })
        
        # Save annotated image
        output_path = output_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"\nSaved annotated image to: {output_path}")
        
        # Save individual JSON file for this image
        json_output = {
            'image_name': img_path.name,
            'image_width': image.shape[1],
            'image_height': image.shape[0],
            'total_components': len(boxes),
            'components': image_detections
        }
        json_path = output_dir / f"{img_path.stem}_detections.json"
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved detection JSON to: {json_path}")
        
        all_detections.append(json_output)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_components = sum(d['total_components'] for d in all_detections)
    
    print(f"\nTotal images processed: {len(all_detections)}")
    print(f"Total components detected: {total_components}")
    print(f"\nAnnotated images saved to: {output_dir}/")
    print(f"Individual JSON files saved to: {output_dir}/")
    
    # Save combined results to file
    combined_results_file = output_dir / 'all_detections.json'
    with open(combined_results_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    print(f"Combined results saved to: {combined_results_file}")
    
    print(f"\n{'='*80}\n")

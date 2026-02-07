from ultralytics import YOLO
from collections import Counter
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

    # Load the trained model from train.py
    model_path = Path('runs/detect/training/weights/best.pt')
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}\n"
            "Run train.py first to generate the model."
        )

    print(f"Using trained model: {model_path}")
    model = YOLO(str(model_path))
    
    # Get class names
    class_names = model.names
    
    # Source and output directories
    # Try proper test set first, fallback to legacy test folder
    test_dirs = [
        Path('./data/training/datasets/aircraft-components/images/test'),  # Proper test set
        Path('./data/testing/images'),  # Legacy location (has data leakage!)
    ]
    source_dir = next((d for d in test_dirs if d.exists()), None)
    if source_dir is None:
        raise FileNotFoundError(
            "No test images found. Run split_data_3way.py first or check paths.\n"
            + "Checked: " + ", ".join(str(d) for d in test_dirs)
        )
    
    print(f"Using test images from: {source_dir}")
    if 'testing/images' in str(source_dir):
        print("⚠️  WARNING: Using legacy test folder - may have data leakage!")
        print("   Run scripts/utils/split_data_3way.py for proper train/val/test split\n")
    
    output_dir = Path('demo/all_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(source_dir.glob('*.png')) + list(source_dir.glob('*.jpg'))
    
    # Predict on validation images (without saving built-in visualizations)
    results = model.predict(
        source=str(source_dir),
        save=False,  # We'll save manually with custom styling
        conf=0.2,
        device=device,
        verbose=False
    )

    all_detections = []

    # Analyze and visualize detections
    for i, (result, img_path) in enumerate(zip(results, image_files)):
        print(f"\n{'='*60}")
        print(f"Validation Image {i+1}/{len(image_files)}: {img_path.name}")
        print('='*60)
        
        # Read original image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_detections = []
        
        if len(result.boxes) > 0:
            # Count by class
            detections = Counter()
            confidences = []
            
            for box_idx, box in enumerate(result.boxes, 1):
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[class_id]
                detections[class_name] += 1
                confidences.append(conf)
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Draw bounding box with smaller thickness (1 instead of 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Add text label with smaller font (0.3 instead of 0.5)
                label = f"{class_name} {conf:.2%}"
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
            
            print(f"Total detections: {len(result.boxes)}")
            print(f"\nBy class:")
            for class_name, count in detections.most_common():
                print(f"  {class_name}: {count}")
            
            print(f"\nConfidence distribution:")
            confidences.sort(reverse=True)
            print(f"  Highest: {confidences[0]:.2%}")
            print(f"  Median: {confidences[len(confidences)//2]:.2%}")
            print(f"  Lowest: {confidences[-1]:.2%}")
            
            # Show top 5 confident detections
            print(f"\nTop 5 detections:")
            sorted_boxes = sorted(result.boxes, key=lambda x: float(x.conf[0]), reverse=True)
            for j, box in enumerate(sorted_boxes[:5]):
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"  {j+1}. {class_names[class_id]}: {conf:.2%}")
        else:
            print("No detections")
        
        # Save annotated image
        output_path = output_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"\nSaved: {output_path}")
        
        # Save individual JSON file
        json_output = {
            'image_name': img_path.name,
            'image_width': image.shape[1],
            'image_height': image.shape[0],
            'total_components': len(result.boxes),
            'components': image_detections
        }
        json_path = output_dir / f"{img_path.stem}_detections.json"
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"Saved: {json_path}")
        
        all_detections.append(json_output)
    
    # Save combined results
    combined_results_file = output_dir / 'all_detections.json'
    with open(combined_results_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Combined results saved to: {combined_results_file}")
    print(f"{'='*60}")
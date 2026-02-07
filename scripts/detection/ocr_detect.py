from ultralytics import YOLO
import torch
import cv2
from pathlib import Path
import easyocr
import numpy as np
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
    
    # Initialize OCR reader
    print("Initializing OCR reader (this may take a moment)...")
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
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
    output_dir = Path('demo/all_ocr_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = list(source_dir.glob('*.png')) + list(source_dir.glob('*.jpg'))
    
    print(f"\n{'='*80}")
    print(f"Processing {len(image_files)} images with YOLO + OCR")
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
            
            # Add padding to bounding box for better OCR
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # Crop the component
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            # Perform OCR on cropped region
            try:
                ocr_results = reader.readtext(cropped, detail=1)
                
                detected_text = []
                for (bbox, text, text_conf) in ocr_results:
                    if text_conf > 0.3:  # Filter low confidence text
                        detected_text.append({
                            'text': text.strip(),
                            'confidence': text_conf
                        })
                
                if detected_text:
                    print(f"\n  Component {box_idx} (conf: {conf:.2%}):")
                    print(f"    Location: ({x1}, {y1}) to ({x2}, {y2})")
                    for txt in detected_text:
                        print(f"    Text: '{txt['text']}' (conf: {txt['confidence']:.2%})")
                    
                    # Draw bounding box and text on image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
                    # Add text label
                    label_text = " | ".join([t['text'] for t in detected_text])
                    label = f"{label_text} ({conf:.2%})"
                    
                    # Create text background
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                    cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                    
                    # Combine all OCR text into a single field
                    ocr_all_text = ' '.join([t['text'] for t in detected_text])
                    
                    image_detections.append({
                        'unique_id': f"{img_path.stem}_component_{box_idx}",
                        'label': class_name,
                        'bbox': {
                            'xmin': x1,
                            'ymin': y1,
                            'xmax': x2,
                            'ymax': y2
                        },
                        'confidence': conf,
                        'ocr_texts': detected_text,
                        'ocr_all_text': ocr_all_text
                    })
                else:
                    # Draw box even if no text detected
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    label = f"No text ({conf:.2%})"
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    
            except Exception as e:
                print(f"  OCR error on component {box_idx}: {e}")
        
        # Save annotated image
        output_path = output_dir / f"ocr_{img_path.name}"
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
        
        all_detections.append({
            'image': img_path.name,
            'total_components': len(boxes),
            'components_with_text': len(image_detections),
            'detections': image_detections
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_components = sum(d['total_components'] for d in all_detections)
    total_with_text = sum(d['components_with_text'] for d in all_detections)
    
    print(f"\nTotal images processed: {len(all_detections)}")
    print(f"Total components detected: {total_components}")
    print(f"Components with text: {total_with_text} ({total_with_text/total_components*100:.1f}%)" if total_components > 0 else "Components with text: 0")
    print(f"\nAnnotated images saved to: {output_dir}/")
    
    # Save detailed results to file
    results_file = output_dir / 'ocr_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    print(f"Detailed results saved to: {results_file}")
    
    print(f"\n{'='*80}\n")

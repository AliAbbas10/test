from ultralytics import YOLO
from pathlib import Path
import torch
import cv2
import json
from collections import Counter


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return list(folder.glob('*.png')) + list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg'))


def _list_labels(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return list(folder.glob('*.txt'))


def _summarize_split(images_dir: Path, labels_dir: Path, split_name: str) -> dict:
    images = _list_images(images_dir)
    labels = _list_labels(labels_dir)

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    images_missing_labels = sorted(image_stems - label_stems)
    labels_missing_images = sorted(label_stems - image_stems)

    class_counts = Counter()
    boxes_total = 0
    max_class_id = -1
    malformed_lines = 0

    for label_path in labels:
        try:
            lines = label_path.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            lines = label_path.read_text(encoding='utf-8', errors='ignore').splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                malformed_lines += 1
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                malformed_lines += 1
                continue
            class_counts[class_id] += 1
            boxes_total += 1
            max_class_id = max(max_class_id, class_id)

    return {
        'name': split_name,
        'images': len(images),
        'labels': len(labels),
        'boxes': boxes_total,
        'max_class_id': max_class_id,
        'malformed_lines': malformed_lines,
        'images_missing_labels': len(images_missing_labels),
        'labels_missing_images': len(labels_missing_images),
        'class_counts': class_counts,
    }


def print_dataset_summary(dataset_path: Path, nc: int | None = None, class_names: dict | None = None) -> None:
    train_images = dataset_path / 'images/train'
    train_labels = dataset_path / 'labels/train'
    val_images = dataset_path / 'images/validation'
    val_labels = dataset_path / 'labels/validation'

    train_summary = _summarize_split(train_images, train_labels, 'train')
    val_summary = _summarize_split(val_images, val_labels, 'validation')

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Dataset root: {dataset_path}")
    print(f"Train: {train_summary['images']} images, {train_summary['labels']} label files, {train_summary['boxes']} boxes")
    print(f"  Missing labels for images: {train_summary['images_missing_labels']}")
    print(f"  Orphan labels without images: {train_summary['labels_missing_images']}")
    print(f"  Max class id in labels: {train_summary['max_class_id']}")
    if train_summary['malformed_lines']:
        print(f"  Malformed label lines: {train_summary['malformed_lines']}")

    print(f"Validation: {val_summary['images']} images, {val_summary['labels']} label files, {val_summary['boxes']} boxes")
    print(f"  Missing labels for images: {val_summary['images_missing_labels']}")
    print(f"  Orphan labels without images: {val_summary['labels_missing_images']}")
    print(f"  Max class id in labels: {val_summary['max_class_id']}")
    if val_summary['malformed_lines']:
        print(f"  Malformed label lines: {val_summary['malformed_lines']}")

    combined_counts = train_summary['class_counts'] + val_summary['class_counts']
    if combined_counts:
        print("\nClass distribution (train+validation, by box count):")
        for class_id, count in combined_counts.most_common(12):
            name = None
            if class_names and class_id in class_names:
                name = class_names[class_id]
            label = f"{class_id}" if name is None else f"{class_id} ({name})"
            print(f"  {label}: {count}")

        if nc is not None:
            missing = [i for i in range(nc) if combined_counts.get(i, 0) == 0]
            if missing:
                preview = ", ".join(str(i) for i in missing[:20])
                suffix = "..." if len(missing) > 20 else ""
                print(f"\nClasses with zero boxes (out of nc={nc}): {preview}{suffix}")

    if nc is not None:
        max_seen = max(train_summary['max_class_id'], val_summary['max_class_id'])
        if max_seen >= nc:
            print("\nWARNING: Found class ids >= nc in labels.")
            print(f"  nc={nc}, max_class_id={max_seen}")
            print("  Fix: update config/train.yaml (nc/names) OR relabel to match.")

    print("=" * 80 + "\n")

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 0
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("No GPU detected, using CPU")

    # Use YOLOv8 Small (better performance than nano)
    modelY = YOLO('models/yolov8s.pt')

    dataset_path = Path('./data/training/datasets/aircraft-components')
    train_yaml = Path('config/train.yaml')
    if not train_yaml.exists():
        raise FileNotFoundError(f"Missing dataset config: {train_yaml}")

    # Parse nc/names from yaml without adding a new dependency
    nc = None
    names = {}
    yaml_text = train_yaml.read_text(encoding='utf-8', errors='ignore').splitlines()
    in_names = False
    for line in yaml_text:
        stripped = line.strip()
        if stripped.startswith('nc:'):
            try:
                nc = int(stripped.split(':', 1)[1].strip())
            except ValueError:
                nc = None
        if stripped.startswith('names:'):
            in_names = True
            continue
        if in_names:
            if not stripped or stripped.startswith('#'):
                continue
            if ':' not in stripped:
                continue
            # Expected format: "0: Connector"
            key, value = stripped.split(':', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            try:
                names[int(key)] = value
            except ValueError:
                pass

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    print_dataset_summary(dataset_path, nc=nc, class_names=names)

    # Train the model with optimized settings
    results = modelY.train(
        data='config/train.yaml',
        epochs=150,              # More epochs
        imgsz=1024,
        batch=4,                 # Smaller batch for larger model
        patience=0,              # Disable early stopping - train all epochs
        device=device,
        project='runs/detect',
        name='training',
        verbose=True,
        plots=False,
        workers=0,               # Fix for Windows multiprocessing
        save_period=10,          # Save checkpoints every 10 epochs
        save=True,               # Save train and val predictions
        save_txt=True,           # Save results to txt files
        save_conf=True,          # Save confidences in labels
        
        # Optimized hyperparameters for small datasets
        lr0=0.001,               # Lower learning rate
        lrf=0.001,               # Final learning rate
        weight_decay=0.001,      # Regularization
        
        # AGGRESSIVE AUGMENTATION (compensates for small dataset)
        hsv_h=0.02,
        hsv_s=0.75,
        hsv_v=0.45,
        degrees=15,              # More rotation
        translate=0.15,          # More translation
        scale=0.7,               # More scale variation
        shear=8,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,          # Copy-paste augmentation
        erasing=0.5,             # Random erasing
    )

    print("\nTraining complete")
    print(f"Run directory: {results.save_dir}")

    # Create validation visualizations with custom styling (quiet)
    print("Creating validation visualizations...")
    
    # Load the best model
    best_model = YOLO(f'{results.save_dir}/weights/best.pt')
    
    # Get class names
    class_names = best_model.names
    
    # Validation images
    val_dir = Path('./data/training/datasets/aircraft-components/images/validation')
    output_dir = Path(results.save_dir) / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all validation images
    image_files = list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg'))
    
    # Run predictions
    val_results = best_model.predict(
        source=str(val_dir),
        save=False,
        conf=0.2,
        device=device,
        verbose=False
    )
    
    all_detections = []
    
    for result, img_path in zip(val_results, image_files):
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_detections = []
        
        if len(result.boxes) > 0:
            for box_idx, box in enumerate(result.boxes, 1):
                # Get bounding box info
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                
                # Draw bounding box with smaller thickness (1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Add label with smaller font (0.3)
                label = f"{class_name} {conf:.2%}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                
                # Add to detections
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
        output_path = output_dir / f"val_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        
        # Save JSON
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
        
        all_detections.append(json_output)
    
    # Save combined results
    combined_file = output_dir / 'all_validation_detections.json'
    with open(combined_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"Validation visualizations: {output_dir}")
    print(f"Validation combined JSON: {combined_file}")
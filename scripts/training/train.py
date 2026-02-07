"""YOLO Aircraft Component Detection Training Script

This script trains a YOLOv8 model to detect aircraft components in technical drawings.
It includes dataset validation, model training with aggressive augmentation strategies,
and automated validation visualization generation.

WORKFLOW:
    1. Run grid_search.py to find optimal hyperparameters (144 combinations × 50 epochs)
    2. Run analyze_grid_search.py to identify best config and save to grid_search_results.csv
    3. Run this script - it automatically loads best hyperparameters and trains for 150 epochs
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import cv2
import json
from collections import Counter
import pandas as pd
import yaml


# ==============================================================================
# DATASET VALIDATION UTILITIES
# ==============================================================================

def _list_images(folder: Path) -> list[Path]:
    """Find all image files in a directory (png, jpg, jpeg)"""
    if not folder.exists():
        return []
    return list(folder.glob('*.png')) + list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg'))


def _list_labels(folder: Path) -> list[Path]:
    """Find all YOLO label files (.txt) in a directory"""
    if not folder.exists():
        return []
    return list(folder.glob('*.txt'))


def _summarize_split(images_dir: Path, labels_dir: Path, split_name: str) -> dict:
    """Analyze a dataset split (train/val) and return statistics
    
    Returns dict with image counts, label counts, bounding box statistics,
    class distribution, and any data integrity issues.
    """
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
    """Print comprehensive dataset statistics and validation report
    
    Analyzes train/validation splits, checks for missing files, validates class IDs,
    shows class distribution, and warns about potential configuration issues.
    """
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


def load_best_hyperparameters() -> dict:
    """Load optimal hyperparameters from grid search results
    
    Returns dict with best hyperparameters, or defaults if grid search hasn't been run.
    Looks for: grid_search_results.csv and the corresponding args.yaml file.
    """
    results_file = Path('grid_search_results.csv')
    
    # Default hyperparameters (if no grid search results exist)
    defaults = {
        'lr0': 0.001,
        'weight_decay': 0.001,
        'hsv_h': 0.02,
        'hsv_s': 0.75,
        'degrees': 15,
        'mosaic': 1.0,
        'source': 'defaults (no grid search results found)'
    }
    
    if not results_file.exists():
        print(f"[WARNING] No grid search results found at {results_file}")
        print("   Using default hyperparameters")
        print("   Run scripts/training/grid_search.py first for better results!\n")
        return defaults
    
    try:
        # Load grid search results
        df = pd.read_csv(results_file)
        
        # Find best experiment by F1 score
        if 'f1_score' in df.columns:
            best_idx = df['f1_score'].idxmax()
            best_exp = int(df.loc[best_idx, 'experiment'])
        else:
            # Fallback to mAP50 if F1 not available
            best_idx = df['best_mAP50'].idxmax()
            best_exp = int(df.loc[best_idx, 'experiment'])
        
        # Load hyperparameters from the best experiment's args.yaml
        args_file = Path(f'runs/detect/grid_search/exp_{best_exp}/args.yaml')
        
        if not args_file.exists():
            print(f"[WARNING] Best experiment args not found: {args_file}")
            print("   Using default hyperparameters\n")
            return defaults
        
        with open(args_file, 'r') as f:
            args = yaml.safe_load(f)
        
        params = {
            'lr0': args.get('lr0', defaults['lr0']),
            'weight_decay': args.get('weight_decay', defaults['weight_decay']),
            'hsv_h': args.get('hsv_h', defaults['hsv_h']),
            'hsv_s': args.get('hsv_s', defaults['hsv_s']),
            'degrees': args.get('degrees', defaults['degrees']),
            'mosaic': args.get('mosaic', defaults['mosaic']),
            'source': f'grid_search/exp_{best_exp}'
        }
        
        print(f"[INFO] Loaded optimal hyperparameters from: grid_search/exp_{best_exp}")
        return params
        
    except Exception as e:
        print(f"[WARNING] Error loading grid search results: {e}")
        print("   Using default hyperparameters\n")
        return defaults


if __name__ == '__main__':
    # ==============================================================================
    # TRAINING WORKFLOW
    # ==============================================================================
    # This script performs full model training (150 epochs) using hyperparameters
    # automatically loaded from grid search results.
    # 
    # Recommended workflow:
    #   1. Run scripts/training/grid_search.py to test 144 hyperparameter combos
    #   2. Run scripts/analysis/analyze_grid_search.py to identify best config
    #   3. Run this script - hyperparameters are loaded automatically from:
    #      - grid_search_results.csv (identifies best experiment)
    #      - runs/detect/grid_search/exp_X/args.yaml (loads hyperparameters)
    #
    # If grid search hasn't been run, sensible defaults are used instead.
    # ==============================================================================
    
    # ==============================================================================
    # INITIALIZATION & SETUP
    # ==============================================================================
    
    # Configure device (GPU if available, CPU otherwise)
    if torch.cuda.is_available():
        device = 0  # Ultralytics accepts integer for GPU index
        device_name = 'cuda:0'  # PyTorch format for .to() method
        print(f"\n[SETUP] GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"[SETUP] CUDA Version: {torch.version.cuda}")
        print(f"[SETUP] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        device_name = 'cpu'
        print("\n[SETUP] No GPU detected, training on CPU")
        print("[WARNING] Training will be VERY slow without GPU!\n")

    # Load YOLOv8 Large pretrained model for final training (better accuracy)
    # Note: Grid search uses yolov8s for speed, this uses yolov8l for performance
    print(f"[SETUP] Loading YOLOv8 Large model for final training...")
    modelY = YOLO('yolov8l.pt')
    
    # Explicitly move model to GPU if available - ultralytics will handle device in train()
    if device != 'cpu':
        print(f"[SETUP] Training will use GPU: {device_name}")
        print(f"[SETUP] Device parameter for training: {device}\n")
    else:
        print(f"[SETUP] Training will use CPU\n")

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

    # Validate dataset integrity before training
    print_dataset_summary(dataset_path, nc=nc, class_names=names)

    # ==============================================================================
    # LOAD OPTIMAL HYPERPARAMETERS FROM GRID SEARCH
    # ==============================================================================
    # Automatically loads best hyperparameters from grid_search_results.csv
    # Falls back to sensible defaults if grid search hasn't been run
    
    print("\n" + "=" * 80)
    print("LOADING HYPERPARAMETERS")
    print("=" * 80)
    best_params = load_best_hyperparameters()
    print(f"\nSource: {best_params['source']}")
    print(f"Learning rate:    {best_params['lr0']}")
    print(f"Weight decay:     {best_params['weight_decay']}")
    print(f"HSV Hue:          {best_params['hsv_h']}")
    print(f"HSV Saturation:   {best_params['hsv_s']}")
    print(f"Rotation degrees: {best_params['degrees']}")
    print(f"Mosaic prob:      {best_params['mosaic']}")
    print("=" * 80 + "\n")

    # ==============================================================================
    # MODEL TRAINING
    # ==============================================================================
    # Training strategy for small datasets:
    # - Aggressive augmentation to increase effective dataset size
    # - No early stopping to ensure full learning
    # - Regular checkpoints for model selection
    # - Hyperparameters automatically loaded from grid search results
    # ==============================================================================
    
    results = modelY.train(
        # Basic training configuration
        data='config/train.yaml',
        epochs=150,              # Full training (grid search uses 50)
        imgsz=1024,
        batch=4,
        patience=0,              # No early stopping - train all epochs
        device=device,
        project='runs/detect',
        name='training',
        verbose=True,
        plots=False,
        workers=0,               # Windows multiprocessing compatibility
        
        # Checkpoint and output settings
        save_period=10,          # Save model every 10 epochs
        save=True,
        save_txt=True,
        save_conf=True,
        
        # Optimized hyperparameters (automatically loaded from grid search)
        lr0=best_params['lr0'],          # Initial learning rate
        lrf=0.001,                        # Final learning rate (no decay)
        weight_decay=best_params['weight_decay'],  # L2 regularization
        
        # Aggressive augmentation strategy to expand limited training data
        # Color space augmentation (grid-search optimized)
        hsv_h=best_params['hsv_h'],      # Hue variation
        hsv_s=best_params['hsv_s'],      # Saturation variation
        hsv_v=0.45,                       # Value/brightness variation (fixed)
        
        # Geometric augmentation (grid-search optimized)
        degrees=best_params['degrees'],   # Random rotation
        translate=0.15,                   # Random translation ±15% (fixed)
        scale=0.7,                        # Random scale 30-100% (fixed)
        shear=8,                          # Random shear ±8° (fixed)
        perspective=0.001,                # Perspective transformation (fixed)
        
        # Flip augmentation (fixed)
        flipud=0.5,                       # Vertical flip probability
        fliplr=0.5,                       # Horizontal flip probability
        
        # Advanced augmentation techniques
        mosaic=best_params['mosaic'],     # Mosaic augmentation (grid-search optimized)
        mixup=0.15,                       # MixUp augmentation probability (fixed)
        copy_paste=0.1,                   # Copy-paste augmentation probability (fixed)
        erasing=0.5,                      # Random erasing probability (fixed)
    )

    # ==============================================================================
    # VALIDATION VISUALIZATION
    # ==============================================================================
    # Generate annotated images and JSON output files for all validation images
    # to visually inspect model performance
    
    print(f"\n[TRAINING COMPLETE] Results saved to: {results.save_dir}")
    print("[VALIDATION] Generating visualizations and detection outputs...")
    
    # Load best performing model from training
    best_model = YOLO(f'{results.save_dir}/weights/best.pt')
    class_names = best_model.names
    
    # Set up validation input/output paths
    val_dir = Path('./data/training/datasets/aircraft-components/images/validation')
    output_dir = Path(results.save_dir) / 'validation_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg'))
    
    # Run inference on all validation images
    val_results = best_model.predict(
        source=str(val_dir),
        save=False,
        conf=0.2,                # 20% confidence threshold
        device=device,
        verbose=False
    )
    
    all_detections = []
    
    # Process each validation image result
    for result, img_path in zip(val_results, image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_detections = []
        
        # Draw bounding boxes and collect detection data
        if len(result.boxes) > 0:
            for box_idx, box in enumerate(result.boxes, 1):
                # Extract detection parameters
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                
                # Draw bounding box (green, thin line)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Draw label background and text (compact style)
                label = f"{class_name} {conf:.2%}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                
                # Store detection metadata
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
        
        # Save annotated image with bounding boxes
        output_path = output_dir / f"val_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        
        # Save per-image detection JSON
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
    
    # Save combined detections for all validation images
    combined_file = output_dir / 'all_validation_detections.json'
    with open(combined_file, 'w') as f:
        json.dump(all_detections, f, indent=2)
    
    print(f"\n[COMPLETE] Annotated images saved to: {output_dir}")
    print(f"[COMPLETE] Combined detections saved to: {combined_file}\n")
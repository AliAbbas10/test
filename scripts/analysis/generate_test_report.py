import json
from pathlib import Path
from collections import Counter, defaultdict
import statistics
import argparse
import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes
    box format: {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
    """
    x1_min, y1_min = box1['xmin'], box1['ymin']
    x1_max, y1_max = box1['xmax'], box1['ymax']
    x2_min, y2_min = box2['xmin'], box2['ymin']
    x2_max, y2_max = box2['xmax'], box2['ymax']
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def load_ground_truth_labels(test_images_dir, class_names):
    """Load ground truth labels for test images
    Returns: dict mapping image_name -> list of {'class_id': int, 'class_name': str, 'bbox': {...}}
    """
    test_images_path = Path(test_images_dir)
    labels_path = test_images_path.parent.parent / 'labels' / 'test'
    
    # Try alternate locations if not found
    if not labels_path.exists():
        labels_path = test_images_path.parent / 'labels'
    if not labels_path.exists():
        return None
    
    ground_truth = {}
    
    for img_path in test_images_path.glob('*.png'):
        label_file = labels_path / (img_path.stem + '.txt')
        if not label_file.exists():
            # Try jpg
            for ext in ['.jpg', '.jpeg']:
                alt_path = test_images_path / (img_path.stem + ext)
                if alt_path.exists():
                    label_file = labels_path / (img_path.stem + '.txt')
                    break
        
        if not label_file.exists():
            continue
        
        # Load image dimensions to convert normalized coords
        try:
            from PIL import Image
            img = Image.open(img_path)
            img_width, img_height = img.size
        except:
            # Default size if can't load image
            img_width, img_height = 1920, 1080
        
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert from normalized YOLO format to pixel coordinates
                    xmin = int((x_center - width/2) * img_width)
                    ymin = int((y_center - height/2) * img_height)
                    xmax = int((x_center + width/2) * img_width)
                    ymax = int((y_center + height/2) * img_height)
                    
                    labels.append({
                        'class_id': class_id,
                        'class_name': class_names.get(class_id, f'class_{class_id}'),
                        'bbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
                    })
        
        ground_truth[img_path.name] = labels
    
    return ground_truth

def calculate_metrics_from_detections(all_detections, ground_truth, iou_threshold=0.5):
    """Calculate precision, recall, F1 per class and overall from detections
    Returns: dict with 'per_class' and 'overall' metrics
    """
    if ground_truth is None:
        return None
    
    # Initialize counters per class
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0})
    overall_tp, overall_fp, overall_fn = 0, 0, 0
    
    for detection in all_detections:
        img_name = detection['image_name']
        predictions = detection['components']
        gt_labels = ground_truth.get(img_name, [])
        
        # Track which ground truth boxes have been matched
        matched_gt = set()
        
        # Sort predictions by confidence (highest first)
        predictions_sorted = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Match predictions to ground truth
        for pred in predictions_sorted:
            pred_class = pred['label']
            pred_bbox = pred['bbox']
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx, gt in enumerate(gt_labels):
                if gt_idx in matched_gt:
                    continue
                if gt['class_name'] != pred_class:
                    continue
                
                iou = calculate_iou(pred_bbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True Positive
                class_stats[pred_class]['tp'] += 1
                overall_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                # False Positive
                class_stats[pred_class]['fp'] += 1
                overall_fp += 1
        
        # Unmatched ground truth boxes are False Negatives
        for gt_idx, gt in enumerate(gt_labels):
            gt_class = gt['class_name']
            class_stats[gt_class]['gt_count'] += 1
            
            if gt_idx not in matched_gt:
                class_stats[gt_class]['fn'] += 1
                overall_fn += 1
    
    # Calculate per-class metrics
    per_class_metrics = []
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics.append({
            'class_name': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': tp + fn  # Total ground truth instances
        })
    
    # Calculate overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    return {
        'per_class': per_class_metrics,
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn
        }
    }

def get_class_names():
    """Load class names from config file"""
    config_path = Path('config/train.yaml')
    if not config_path.exists():
        return {}
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('names', {})


def load_model_and_validate(model_path: Path, data_yaml: str = "config/train.yaml", imgsz: int = 1024, conf: float = 0.25):
    """Run YOLO validation and return metrics"""
    try:
        from ultralytics import YOLO
        
        model = YOLO(str(model_path))
        
        # Run validation
        print("\nRunning validation on validation set...")
        metrics = model.val(
            data=data_yaml,
            imgsz=imgsz,
            conf=conf,
            plots=False,
            save_json=False,
            verbose=False,
            workers=0,
        )
        
        # Extract metrics
        def _format_float(x):
            try:
                return float(x)
            except:
                return 0.0
        
        overall_p = _format_float(getattr(metrics.box, "mp", 0.0))
        overall_r = _format_float(getattr(metrics.box, "mr", 0.0))
        overall_map50 = _format_float(getattr(metrics.box, "map50", 0.0))
        overall_map = _format_float(getattr(metrics.box, "map", 0.0))
        overall_f1 = (2.0 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0
        
        # Get model names
        names = getattr(model, "names", {}) or {}
        
        # Per-class metrics
        import numpy as np
        
        def _to_1d_array(x):
            if x is None:
                return np.array([])
            if isinstance(x, (list, tuple)):
                return np.array(x, dtype=float)
            if hasattr(x, "shape"):
                arr = np.asarray(x, dtype=float)
                return arr.reshape(-1)
            return np.array([])
        
        per_class_p = _to_1d_array(getattr(metrics.box, "p", None))
        per_class_r = _to_1d_array(getattr(metrics.box, "r", None))
        per_class_map50 = _to_1d_array(getattr(metrics.box, "ap50", None))
        
        per_class_metrics = []
        if per_class_p.size and per_class_r.size:
            n = min(per_class_p.size, per_class_r.size)
            for class_id in range(n):
                p = float(per_class_p[class_id])
                r = float(per_class_r[class_id])
                f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
                m50 = float(per_class_map50[class_id]) if per_class_map50.size > class_id else 0.0
                name = names.get(class_id, str(class_id))
                per_class_metrics.append({
                    'class_id': class_id,
                    'class_name': name,
                    'precision': p,
                    'recall': r,
                    'f1': f1,
                    'map50': m50
                })
        
        return {
            'precision': overall_p,
            'recall': overall_r,
            'f1': overall_f1,
            'map50': overall_map50,
            'map50_95': overall_map,
            'per_class': per_class_metrics
        }
    except Exception as e:
        print(f"Warning: Could not compute validation metrics: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive test report with validation metrics")
    parser.add_argument("--validate", action="store_true", help="Include validation metrics (requires validation set)")
    parser.add_argument("--data", type=str, default="config/train.yaml", help="Dataset YAML for validation")
    parser.add_argument("--imgsz", type=int, default=1024, help="Validation image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for validation")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching predictions to ground truth")
    args = parser.parse_args()

    # Load all detection results
    results_file = Path('demo/all_analysis/all_detections.json')
    if not results_file.exists():
        print(f"Error: {results_file} not found. Run test.py first.")
        return
    
    with open(results_file, 'r') as f:
        all_detections = json.load(f)

    # Get model info dynamically
    model_path = Path('models/best_trained_model.pt')
    if not model_path.exists():
        model_path = Path('runs/detect/grid_search/exp_18/weights/best.pt')

    print("="*80)
    print("COMPREHENSIVE TEST REPORT")
    print("="*80)
    print(f"Model: {model_path}")
    print("="*80)
    
    # Load class names
    class_names = get_class_names()
    
    # Load ground truth and calculate test metrics
    test_images_dir = None
    for potential_dir in [
        Path('data/training/datasets/aircraft-components/images/test'),
        Path('data/testing/images/test'),
        Path('data/testing/images')
    ]:
        if potential_dir.exists():
            test_images_dir = potential_dir
            break
    
    test_metrics = None
    ground_truth = None
    if test_images_dir:
        ground_truth = load_ground_truth_labels(test_images_dir, class_names)
        if ground_truth:
            test_metrics = calculate_metrics_from_detections(all_detections, ground_truth, args.iou)
    
    # Display test set metrics (ground truth comparison)
    if test_metrics:
        print("\nTEST SET METRICS (IoU @ {:.2f}):".format(args.iou))
        print("  Class-Agnostic (Overall):")
        print(f"    Precision:   {test_metrics['overall']['precision']:.4f}")
        print(f"    Recall:      {test_metrics['overall']['recall']:.4f}")
        print(f"    F1 Score:    {test_metrics['overall']['f1']:.4f}")
        print(f"    TP: {test_metrics['overall']['tp']}, FP: {test_metrics['overall']['fp']}, FN: {test_metrics['overall']['fn']}")
    
    # Validation metrics (if requested)
    validation_metrics = None
    if args.validate and model_path.exists():
        validation_metrics = load_model_and_validate(model_path, args.data, args.imgsz, args.conf)
        if validation_metrics:
            print("\nVALIDATION SET METRICS (on validation set):")
            print("  Class-Agnostic (Overall):")
            print(f"    Precision:   {validation_metrics['precision']:.4f}")
            print(f"    Recall:      {validation_metrics['recall']:.4f}")
            print(f"    F1 Score:    {validation_metrics['f1']:.4f}")
            print(f"    mAP@50:      {validation_metrics['map50']:.4f}")
            print(f"    mAP@50-95:   {validation_metrics['map50_95']:.4f}")

    # Overall statistics
    total_images = len(all_detections)
    total_detections = sum(img['total_components'] for img in all_detections)
    images_with_detections = sum(1 for img in all_detections if img['total_components'] > 0)
    images_without_detections = total_images - images_with_detections

    print(f"\nTEST SET DETECTION STATISTICS:")
    print(f"  Total test images:           {total_images}")
    print(f"  Images with detections:      {images_with_detections} ({images_with_detections/total_images*100:.1f}%)")
    print(f"  Images without detections:   {images_without_detections} ({images_without_detections/total_images*100:.1f}%)")
    print(f"  Total components detected:   {total_detections}")
    print(f"  Avg components per image:    {total_detections/total_images:.2f}")

    # Detection count distribution
    detection_counts = [img['total_components'] for img in all_detections]
    if detection_counts:
        print(f"\nDETECTION DISTRIBUTION:")
        print(f"  Min detections per image:    {min(detection_counts)}")
        print(f"  Max detections per image:    {max(detection_counts)}")
        print(f"  Median detections per image: {statistics.median(detection_counts):.1f}")
        print(f"  Mean detections per image:   {statistics.mean(detection_counts):.2f}")
        print(f"  Std deviation:               {statistics.stdev(detection_counts) if len(detection_counts) > 1 else 0:.2f}")

    # Class distribution
    all_classes = Counter()
    all_confidences = []

    for img in all_detections:
        for component in img['components']:
            all_classes[component['label']] += 1
            all_confidences.append(component['confidence'])

    print(f"\nCLASS DISTRIBUTION (Top 15):")
    for class_name, count in all_classes.most_common(15):
        percentage = count / total_detections * 100
        print(f"  {class_name:20s}: {count:4d} ({percentage:5.2f}%)")

    if len(all_classes) > 15:
        print(f"  ... and {len(all_classes) - 15} more classes")

    # Confidence statistics
    if all_confidences:
        all_confidences.sort(reverse=True)
        print(f"\nCONFIDENCE STATISTICS:")
        print(f"  Mean confidence:    {statistics.mean(all_confidences):.2%}")
        print(f"  Median confidence:  {statistics.median(all_confidences):.2%}")
        print(f"  Highest confidence: {all_confidences[0]:.2%}")
        print(f"  Lowest confidence:  {all_confidences[-1]:.2%}")
        print(f"  95th percentile:    {all_confidences[int(len(all_confidences)*0.05)]:.2%}")
        print(f"  5th percentile:     {all_confidences[int(len(all_confidences)*0.95)]:.2%}")

    # Top 10 images by detection count
    print(f"\nTOP 10 IMAGES BY DETECTION COUNT:")
    sorted_images = sorted(all_detections, key=lambda x: x['total_components'], reverse=True)[:10]
    for i, img in enumerate(sorted_images, 1):
        print(f"  {i:2d}. {img['image_name']:30s}: {img['total_components']:3d} components")

    # High confidence detections (>80%)
    high_conf_detections = []
    for img in all_detections:
        for comp in img['components']:
            if comp['confidence'] > 0.8:
                high_conf_detections.append((img['image_name'], comp['label'], comp['confidence']))

    print(f"\nHIGH CONFIDENCE DETECTIONS (>80%): {len(high_conf_detections)} total")
    if high_conf_detections:
        high_conf_detections.sort(key=lambda x: x[2], reverse=True)
        print(f"Top 10:")
        for i, (img_name, label, conf) in enumerate(high_conf_detections[:10], 1):
            print(f"  {i:2d}. {label:15s} in {img_name:25s}: {conf:.2%}")

    # Per-class validation metrics (if available)
    if validation_metrics and validation_metrics['per_class']:
        print(f"\nPER-CLASS VALIDATION METRICS (sorted by F1):")
        per_class = sorted(validation_metrics['per_class'], key=lambda x: x['f1'], reverse=True)
        print(f"{'ID':>3}  {'Class':<14}  {'P':>7}  {'R':>7}  {'F1':>7}  {'mAP50':>7}")
        print("-" * 60)
        for cls in per_class[:15]:  # Top 15
            print(f"{cls['class_id']:>3}  {cls['class_name']:<14.14}  "
                  f"{cls['precision']:7.3f}  {cls['recall']:7.3f}  "
                  f"{cls['f1']:7.3f}  {cls['map50']:7.3f}")
        if len(per_class) > 15:
            print(f"  ... and {len(per_class) - 15} more classes")
    
    # Per-class test metrics (if available)
    if test_metrics and test_metrics['per_class']:
        print(f"\nPER-CLASS TEST SET METRICS (sorted by F1, IoU @ {args.iou}):")
        per_class = sorted(test_metrics['per_class'], key=lambda x: x['f1'], reverse=True)
        print(f"{'Class':<14}  {'P':>7}  {'R':>7}  {'F1':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'Support':>7}")
        print("-" * 80)
        for cls in per_class:
            print(f"{cls['class_name']:<14.14}  "
                  f"{cls['precision']:7.3f}  {cls['recall']:7.3f}  "
                  f"{cls['f1']:7.3f}  {cls['tp']:5d}  "
                  f"{cls['fp']:5d}  {cls['fn']:5d}  {cls['support']:7d}")

    print("\n" + "="*80)
    print(f"RESULTS SAVED TO: demo/all_analysis/")
    print(f"  - Annotated images: detected_*.png")
    print(f"  - Detection JSONs: *_detections.json")
    print(f"  - Combined JSON: all_detections.json")
    print("="*80)
    
    if not test_metrics:
        print("\nNote: Ground truth labels not found. Per-class metrics require:")
        print("  - Labels in data/training/datasets/aircraft-components/labels/test/")
        print("  - or data/testing/labels/")
    
    if not args.validate:
        print("\nTip: Run with --validate flag to include validation set metrics:")
        print("  python scripts/analysis/generate_test_report.py --validate")
    print()

if __name__ == "__main__":
    main()

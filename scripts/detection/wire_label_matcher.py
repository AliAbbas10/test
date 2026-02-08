import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import easyocr


# ----------------------------
# Normalization and filtering helpers
# ----------------------------
def normalize_label(text: str) -> str:
    """
    Normalize OCR output to improve matching for wiring labels.
    Standardizes format and fixes common OCR artifacts.
    """
    if not text:
        return ""
    
    text = text.strip().upper()
    
    # Remove whitespace within tokens
    text = re.sub(r'\s+', '', text)
    
    # Keep only alphanumeric and common label characters
    text = re.sub(r'[^A-Z0-9\-_/\.\+]', '', text)
    
    # Fix common OCR errors (be conservative to avoid over-correction)
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('|', '1')  # pipe to 1
    
    # Only fix O->0 at the end of strings or before numbers (more likely to be digit)
    text = re.sub(r'O(\d)', r'0\1', text)  # O before digit -> 0
    text = re.sub(r'(\d)O(\d)', r'\g<1>0\2', text)  # O between digits -> 0
    
    # Fix I->1 when surrounded by digits
    text = re.sub(r'(\d)I(\d)', r'\g<1>1\2', text)  # I between digits -> 1
    
    return text


def is_valid_wire_label(text: str) -> bool:
    """
    More permissive heuristic to determine if text looks like a valid wire/component label.
    """
    if not text or len(text) < 1:  # Allow single characters
        return False
    
    # Skip common words that aren't labels
    blacklist = {
        'GROUND', 'GND', 'BUS', 'RELAY', 'SWITCH', 'INVERTER', 'CONNECTOR', 
        'VOLTS', 'VAC', 'VDC', 'POWER', 'PWR', 'INPUT', 'OUTPUT',
        'ON', 'OFF', 'OPEN', 'CLOSE', 'HOT', 'COM', 'NEUTRAL', 'TESTSWITCH',
        'TESTSWTTCH', '1TESTSWITCH', '2TESTSWITCH', '3TESTSWITCH', 'THE', 'AND', 'OR'
    }
    if text in blacklist:
        return False
    
    # Skip pure letters without numbers (unless short component names)
    if text.isalpha() and len(text) > 3:
        return False
        
    # Skip pure numbers unless they look like component values
    if text.isdigit() and len(text) > 4:
        return False
    
    # Must contain alphanumeric characters
    if not any(c.isalnum() for c in text):
        return False
    
    # More permissive patterns - accept most alphanumeric combinations
    patterns = [
        r'^[A-Z]\d+[A-Z]*$',                    # J1, X2A, etc.
        r'^[A-Z]{1,4}\d+[A-Z]*$',               # CB1, SW12, ABCD1, etc.
        r'^[A-Z]+\d+[-_][A-Z\d]+$',             # X1A-CB, J9-12, etc.
        r'^\d+[A-Z]+\d*$',                      # 12A, 115V, etc.
        r'^[A-Z]\d+[-_/\.]\d+[A-Z]*$',          # J1-2, X3/4, etc.
        r'^W\d+[A-Z]*\d*[-_]?[A-Z]*$',         # Wire labels like W14420, W11820-CB
        r'^P\d+[A-Z]*$',                        # Pin labels like P281, P33
        r'^[A-Z]+\d+[A-Z]*$',                   # General component labels
        r'^\d+[A-Z]?\d*$',                      # Simple numeric with optional letter
        r'^[A-Z]{1,3}\d+[A-Z\d]*$',            # Short prefix + numbers
        r'^[A-Z]\d*[A-Z]+\d*$',                 # Mixed letter-number patterns
        r'^\d+$',                               # Pure numbers (component values)
        r'^[A-Z]+$'                             # Pure letters if short (like CB, SW)
    ]
    
    # Accept if matches any pattern OR has mix of letters and numbers
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    
    return (any(re.match(pattern, text) for pattern in patterns) or 
            (has_letter and has_digit) or 
            (has_letter and len(text) <= 3) or 
            (has_digit and len(text) <= 4))


def calculate_bbox_center(bbox_points) -> tuple[float, float]:
    """
    Calculate center point of bounding box.
    EasyOCR returns 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    points = np.array(bbox_points, dtype=np.float32)
    center_x = float(np.mean(points[:, 0]))
    center_y = float(np.mean(points[:, 1]))
    return center_x, center_y


def calculate_bbox_area(bbox_points) -> float:
    """Calculate the area of the bounding box."""
    points = np.array(bbox_points, dtype=np.float32)
    return cv2.contourArea(points)


# ----------------------------
# Image preprocessing
# ----------------------------
def enhance_image_for_ocr(image: np.ndarray):
    """
    Create multiple enhanced versions of the image for better OCR performance.
    Returns list of processed images to try.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    enhanced_versions = []
    
    # Version 1: Basic CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced1 = clahe.apply(gray)
    denoised1 = cv2.bilateralFilter(enhanced1, d=9, sigmaColor=75, sigmaSpace=75)
    enhanced_versions.append(denoised1)
    
    # Version 2: High contrast with morphology
    enhanced2 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    enhanced2 = cv2.morphologyEx(enhanced2, cv2.MORPH_CLOSE, kernel)
    enhanced_versions.append(enhanced2)
    
    # Version 3: Gaussian blur + unsharp mask
    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    enhanced_versions.append(unsharp)
    
    # Version 4: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    enhanced_versions.append(adaptive)
    
    # Version 5: Original with mild enhancement
    mild = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    enhanced_versions.append(mild)
    
    return enhanced_versions


# ----------------------------
# Main processing functions
# ----------------------------
def extract_text_with_ocr(image: np.ndarray, use_gpu: bool = False, min_confidence: float = 0.3):
    """
    Extract text from image using EasyOCR with optimized single-pass processing.
    Returns list of detections with bbox, text, and confidence.
    """
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
    
    # Use the best preprocessing approach
    enhanced_versions = enhance_image_for_ocr(image)
    enhanced_image = enhanced_versions[0]  # Use the first (best) version
    
    # Extract text with optimized parameters for technical diagrams
    ocr_results = reader.readtext(
        enhanced_image, 
        detail=1, 
        paragraph=False,
        width_ths=0.3,      # Lower threshold to catch smaller text
        height_ths=0.3,     # Lower threshold to catch smaller text
        text_threshold=0.3, # Lower threshold for text detection
        low_text=0.2        # Lower threshold for weak text
    )
    
    detections = []
    for bbox, text, confidence in ocr_results:
        if confidence < min_confidence:
            continue
            
        normalized_text = normalize_label(text)
        if not is_valid_wire_label(normalized_text):
            continue
            
        center_x, center_y = calculate_bbox_center(bbox)
        area = calculate_bbox_area(bbox)
        
        detection = {
            'original_text': text,
            'normalized_label': normalized_text,
            'confidence': float(confidence),
            'bbox_points': [[float(x), float(y)] for x, y in bbox],
            'center_position': [center_x, center_y],
            'bbox_area': area
        }
        
        detections.append(detection)
    
    return detections


def find_label_matches(detections: list, min_occurrences: int = 2) -> dict:
    """
    Group detections by normalized label and find matches.
    Returns dictionary of labels with their occurrences.
    """
    label_groups = defaultdict(list)
    
    for detection in detections:
        label = detection['normalized_label']
        label_groups[label].append(detection)
    
    # Keep only labels that appear multiple times
    matches = {
        label: occurrences 
        for label, occurrences in label_groups.items() 
        if len(occurrences) >= min_occurrences
    }
    
    return matches


def create_annotated_image(original_image: np.ndarray, matches: dict, output_path: Path) -> None:
    """
    Create an annotated image showing matched labels with different colors.
    """
    annotated = original_image.copy()
    
    # Generate distinct colors for each label
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
    ]
    
    label_list = list(matches.keys())
    
    for i, (label, occurrences) in enumerate(matches.items()):
        color = colors[i % len(colors)]
        
        for occurrence in occurrences:
            # Draw bounding box
            bbox_points = np.array(occurrence['bbox_points'], dtype=np.int32)
            cv2.polylines(annotated, [bbox_points], isClosed=True, color=color, thickness=2)
            
            # Draw center point
            center = tuple(map(int, occurrence['center_position']))
            cv2.circle(annotated, center, 3, color, -1)
            
            # Add label text
            text_pos = (center[0] + 8, center[1] - 8)
            cv2.putText(
                annotated, 
                label, 
                text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2, 
                cv2.LINE_AA
            )
    
    # Save annotated image
    cv2.imwrite(str(output_path), annotated)


def save_results(matches: dict, output_dir: Path, image_name: str) -> dict:
    """
    Save results to JSON file and return summary statistics.
    """
    # Prepare output data
    output_data = {
        'image_processed': image_name,
        'total_unique_labels': len(matches),
        'total_detections': sum(len(occurrences) for occurrences in matches.values()),
        'matched_labels': {}
    }
    
    for label, occurrences in matches.items():
        output_data['matched_labels'][label] = {
            'count': len(occurrences),
            'positions': []
        }
        
        for occurrence in occurrences:
            position_data = {
                'original_text': occurrence['original_text'],
                'confidence': occurrence['confidence'],
                'center_position': occurrence['center_position'],
                'bbox_points': occurrence['bbox_points'],
                'bbox_area': occurrence['bbox_area']
            }
            output_data['matched_labels'][label]['positions'].append(position_data)
    
    # Save to JSON file
    json_output_path = output_dir / f"{Path(image_name).stem}_label_matches.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_data


# ----------------------------
# Main function
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract and match wire labels from technical diagrams using OCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--image', 
        required=True, 
        help="Path to input image (PNG, JPG, etc.)"
    )
    parser.add_argument(
        '--output_dir', 
        default='wire_label_matches', 
        help="Output directory for results"
    )
    parser.add_argument(
        '--min_confidence', 
        type=float, 
        default=0.3, 
        help="Minimum OCR confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        '--min_occurrences', 
        type=int, 
        default=1, 
        help="Minimum number of occurrences to consider a label as matched (1=show all labels)"
    )
    parser.add_argument(
        '--use_gpu', 
        action='store_true', 
        help="Enable GPU acceleration for OCR (if available)"
    )
    parser.add_argument(
        '--no_annotation', 
        action='store_true', 
        help="Skip creating annotated image"
    )
    
    args = parser.parse_args()
    
    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing image: {image_path}")
    print(f"Output directory: {output_dir}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Extract text detections
    print("Extracting text with OCR...")
    detections = extract_text_with_ocr(
        image, 
        use_gpu=args.use_gpu, 
        min_confidence=args.min_confidence
    )
    
    print(f"Found {len(detections)} valid text detections")
    
    # Find label matches
    matches = find_label_matches(detections, min_occurrences=args.min_occurrences)
    
    print(f"Found {len(matches)} labels with {args.min_occurrences}+ occurrences")
    
    # Save results
    results = save_results(matches, output_dir, image_path.name)
    json_path = output_dir / f"{image_path.stem}_label_matches.json"
    print(f"Saved results to: {json_path}")
    
    # Create annotated image
    if not args.no_annotation and matches:
        annotation_path = output_dir / f"{image_path.stem}_annotated.png"
        create_annotated_image(image, matches, annotation_path)
        print(f"Saved annotated image to: {annotation_path}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total unique matched labels: {results['total_unique_labels']}")
    print(f"  Total matched detections: {results['total_detections']}")
    
    if matches:
        print("\nMatched labels:")
        for label, occurrences in sorted(matches.items()):
            print(f"  {label}: {len(occurrences)} occurrences")


if __name__ == "__main__":
    main()
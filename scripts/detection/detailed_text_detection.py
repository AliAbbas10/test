"""
Detailed Text Detection Script with Image Preprocessing
Detects all text in images with precise coordinates and applies
various preprocessing techniques for optimal OCR results.
"""

import cv2
import numpy as np
import easyocr
import torch
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Tuple
import argparse


class TextDetector:
    """Advanced text detection with image preprocessing"""
    
    def __init__(self, gpu='auto', languages=['en']):
        """Initialize OCR reader"""
        # Auto-detect GPU if requested
        if gpu == 'auto':
            gpu = torch.cuda.is_available()
            if gpu:
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print("Using GPU for OCR acceleration")
            else:
                print("No GPU detected, using CPU")
        
        print("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.use_gpu = gpu
        print("OCR reader initialized successfully")
    
    def preprocess_image(self, image: np.ndarray, methods: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Apply various preprocessing methods to improve OCR accuracy
        
        Args:
            image: Input image (BGR or grayscale)
            methods: List of preprocessing methods to apply
                    Options: 'grayscale', 'denoise', 'clahe', 'threshold_otsu',
                            'threshold_adaptive', 'sharpen', 'morphology', 'deskew'
        
        Returns:
            Dictionary of preprocessed images with method names as keys
        """
        if methods is None:
            methods = ['grayscale', 'denoise', 'clahe', 'threshold_adaptive']
        
        processed_images = {}
        current = image.copy()
        
        # Always start with grayscale conversion
        if len(current.shape) == 3:
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        else:
            gray = current.copy()
        
        processed_images['original'] = image
        processed_images['grayscale'] = gray
        
        # Denoising
        if 'denoise' in methods:
            # Bilateral filter preserves edges while removing noise
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            processed_images['denoised'] = denoised
            current = denoised
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if 'clahe' in methods:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(current if 'denoise' in methods else gray)
            processed_images['clahe'] = clahe_img
            current = clahe_img
        
        # Sharpening
        if 'sharpen' in methods:
            kernel_sharpen = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
            sharpened = cv2.filter2D(current, -1, kernel_sharpen)
            processed_images['sharpened'] = sharpened
            current = sharpened
        
        # Otsu's Thresholding
        if 'threshold_otsu' in methods:
            _, otsu = cv2.threshold(current, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images['otsu'] = otsu
        
        # Adaptive Thresholding (better for varying lighting)
        if 'threshold_adaptive' in methods:
            adaptive = cv2.adaptiveThreshold(
                current, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
            processed_images['adaptive'] = adaptive
        
        # Morphological operations
        if 'morphology' in methods:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # Closing: dilation followed by erosion (fills small holes)
            closing = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel)
            processed_images['morphology'] = closing
        
        # Deskewing
        if 'deskew' in methods:
            deskewed = self.deskew_image(current)
            processed_images['deskewed'] = deskewed
        
        # Advanced contrast enhancement
        if 'enhance_contrast' in methods:
            # Normalize and enhance
            normalized = cv2.normalize(current, None, 0, 255, cv2.NORM_MINMAX)
            processed_images['contrast_enhanced'] = normalized
        
        # Unsharp masking for text clarity
        if 'unsharp_mask' in methods:
            gaussian = cv2.GaussianBlur(current, (0, 0), 2.0)
            unsharp = cv2.addWeighted(current, 1.5, gaussian, -0.5, 0)
            processed_images['unsharp'] = unsharp
        
        # Multiple scale processing
        if 'multiscale' in methods:
            # Upscale for better small text detection
            upscaled = cv2.resize(current, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            processed_images['multiscale'] = upscaled
        
        return processed_images
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct skew in the image"""
        # Find all white pixels
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
        
        # Calculate the angle of skew
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate the image to deskew it
        if abs(angle) > 0.5:  # Only deskew if angle is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
    
    def detect_text(self, image: np.ndarray, detail: int = 1, paragraph: bool = False) -> List[Tuple]:
        """
        Detect text in the image with enhanced parameters
        
        Args:
            image: Input image (can be preprocessed)
            detail: OCR detail level (0=low, 1=medium/default, 2=high)
            paragraph: Combine text into paragraphs
        
        Returns:
            List of tuples: (bounding_box, text, confidence)
            bounding_box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # Enhanced OCR parameters for better accuracy
        results = self.reader.readtext(
            image,
            detail=detail,
            paragraph=paragraph,
            min_size=10,  # Minimum text box size
            text_threshold=0.6,  # Text detection threshold (lower = more sensitive)
            low_text=0.3,  # Link threshold (lower = more sensitive)
            link_threshold=0.3,  # Link threshold for text connection
            canvas_size=2560,  # Larger canvas for better detection
            mag_ratio=1.5,  # Magnification ratio
            slope_ths=0.1,  # Slope threshold for text line
            ycenter_ths=0.5,  # Y-center threshold
            height_ths=0.5,  # Height threshold
            width_ths=0.5,  # Width threshold
            add_margin=0.1,  # Add margin to bounding box
        )
        return results
    
    def process_image_with_best_preprocessing(
        self,
        image_path: str,
        output_dir: str = None,
        save_preprocessed: bool = True
    ) -> Dict:
        """
        Process an image with multiple preprocessing methods and select best results
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results and preprocessed images
            save_preprocessed: Whether to save preprocessed images
        
        Returns:
            Dictionary with detection results and metadata
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_name = Path(image_path).stem
        
        # Apply different preprocessing combinations (expanded for better accuracy)
        preprocessing_configs = [
            {
                'name': 'original',
                'methods': []
            },
            {
                'name': 'basic',
                'methods': ['grayscale', 'denoise']
            },
            {
                'name': 'enhanced',
                'methods': ['grayscale', 'denoise', 'clahe']
            },
            {
                'name': 'enhanced_contrast',
                'methods': ['grayscale', 'denoise', 'clahe', 'enhance_contrast']
            },
            {
                'name': 'unsharp_enhanced',
                'methods': ['grayscale', 'denoise', 'unsharp_mask', 'clahe']
            },
            {
                'name': 'threshold_adaptive',
                'methods': ['grayscale', 'denoise', 'clahe', 'threshold_adaptive']
            },
            {
                'name': 'threshold_otsu',
                'methods': ['grayscale', 'denoise', 'threshold_otsu']
            },
            {
                'name': 'sharpened',
                'methods': ['grayscale', 'denoise', 'sharpen', 'threshold_adaptive']
            },
            {
                'name': 'multiscale_enhanced',
                'methods': ['grayscale', 'denoise', 'clahe', 'multiscale']
            },
            {
                'name': 'premium',
                'methods': ['grayscale', 'denoise', 'unsharp_mask', 'clahe', 'enhance_contrast']
            }
        ]
        
        all_results = []
        
        print(f"\nProcessing: {img_name}")
        print("=" * 60)
        
        for config in preprocessing_configs:
            print(f"\nTrying preprocessing: {config['name']}")
            
            # Get preprocessed images
            processed = self.preprocess_image(image, config['methods'])
            
            # Use the final processed image for OCR
            if config['methods']:
                # Get the last processed version
                final_key = list(processed.keys())[-1]
                ocr_image = processed[final_key]
            else:
                ocr_image = image
            
            # Detect text with high detail level
            detections = self.detect_text(ocr_image, detail=1)
            
            # Calculate average confidence
            avg_confidence = 0
            if detections:
                avg_confidence = sum(det[2] for det in detections) / len(detections)
            
            result = {
                'preprocessing': config['name'],
                'methods': config['methods'],
                'num_detections': len(detections),
                'avg_confidence': float(avg_confidence),
                'detections': [
                    {
                        'bbox': [[int(x), int(y)] for x, y in (bbox.tolist() if isinstance(bbox, np.ndarray) else bbox)],
                        'text': text,
                        'confidence': float(conf)
                    }
                    for bbox, text, conf in detections
                ]
            }
            
            all_results.append(result)
            
            print(f"  Found {len(detections)} text regions")
            print(f"  Average confidence: {avg_confidence:.3f}")
            
            # Save preprocessed image if requested
            if save_preprocessed and output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True, parents=True)
                
                if config['methods']:
                    final_key = list(processed.keys())[-1]
                    save_img = processed[final_key]
                    save_path = output_path / f"{img_name}_{config['name']}.png"
                    cv2.imwrite(str(save_path), save_img)
        
        # Select best preprocessing method (highest average confidence)
        best_result = max(all_results, key=lambda x: x['avg_confidence'])
        
        print("\n" + "=" * 60)
        print(f"Best preprocessing: {best_result['preprocessing']}")
        print(f"Total detections: {best_result['num_detections']}")
        print(f"Average confidence: {best_result['avg_confidence']:.3f}")
        
        # Create comprehensive result
        final_result = {
            'image_name': img_name,
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'image_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            },
            'best_preprocessing': best_result['preprocessing'],
            'best_result': best_result,
            'all_results': all_results
        }
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            # Save JSON results
            json_path = output_path / f"{img_name}_text_detection.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {json_path}")
            
            # Save annotated image with best results
            annotated = self.annotate_image(image, best_result['detections'])
            annotated_path = output_path / f"{img_name}_annotated.png"
            cv2.imwrite(str(annotated_path), annotated)
            print(f"Annotated image saved to: {annotated_path}")
        
        return final_result
    
    def annotate_image(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and text on image"""
        annotated = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            text = det['text']
            confidence = det['confidence']
            
            # Convert bbox to integer points
            points = np.array(bbox, dtype=np.int32)
            
            # Draw polygon
            cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
            
            # Draw text label
            label = f"{text} ({confidence:.2f})"
            # Position label above the bbox
            text_pos = (points[0][0], max(points[0][1] - 10, 20))
            
            # Add background for text
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (text_pos[0], text_pos[1] - label_h - 5),
                (text_pos[0] + label_w, text_pos[1] + 5),
                (0, 255, 0),
                -1
            )
            
            cv2.putText(
                annotated,
                label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
        
        return annotated


def main():
    parser = argparse.ArgumentParser(
        description='Detect text in images with advanced preprocessing'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/text_detection',
        help='Directory to save results (default: ./output/text_detection)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='Use GPU for OCR if available (default: auto-detect)'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Force CPU usage even if GPU is available'
    )
    parser.add_argument(
        '--no-save-preprocessed',
        action='store_true',
        help='Do not save preprocessed images'
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Initialize detector with GPU auto-detection
    use_gpu = 'auto' if not args.no_gpu else False
    detector = TextDetector(gpu=use_gpu)
    
    # Process image
    result = detector.process_image_with_best_preprocessing(
        args.image_path,
        output_dir=args.output_dir,
        save_preprocessed=not args.no_save_preprocessed
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    
    best = result['best_result']
    print(f"\nBest preprocessing method: {best['preprocessing']}")
    print(f"Number of text detections: {best['num_detections']}")
    print(f"Average confidence: {best['avg_confidence']:.3f}")
    
    print("\nDetected text:")
    for i, det in enumerate(best['detections'], 1):
        print(f"\n{i}. Text: '{det['text']}'")
        print(f"   Confidence: {det['confidence']:.3f}")
        print(f"   Bounding box: {det['bbox']}")


if __name__ == '__main__':
    main()

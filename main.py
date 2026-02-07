"""
Main pipeline for aircraft component detection, classification, and OCR

This script orchestrates:
1. YOLO object detection with bounding boxes
2. Component classification
3. OCR text extraction
4. Comprehensive JSON output generation
"""

import subprocess
import json
from pathlib import Path
import sys


class ComponentDetectionPipeline:
    def __init__(self):
        self.python_executable = sys.executable
        self.output_dir = Path('./demo')
        self.images_dir = Path('./data/testing/images')
        self.results_dir = Path('./demo/complete_results')
    
    def execute_python(self, script_path, args=None):
        """Execute a Python script and return the result"""
        if args is None:
            args = []
        
        print(f"\n{'='*80}")
        print(f"Executing: {script_path}")
        print(f"{'='*80}\n")
        
        result = subprocess.run(
            [self.python_executable, script_path] + args,
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Script exited with code {result.returncode}")
        
        return result
    
    def run_yolo_detection(self):
        """Step 1: Run YOLO detection with bounding boxes"""
        print("\nStep 1: Running YOLO Detection with Bounding Boxes...\n")
        self.execute_python("scripts/detection/detect_json.py")
    
    def run_yolo_with_ocr(self):
        """Step 2: Run YOLO detection with OCR"""
        print("\nStep 2: Running YOLO Detection with OCR...\n")
        self.execute_python("scripts/detection/ocr_detect.py")
    
    def run_validation_analysis(self):
        """Step 3: Run validation analysis"""
        print("\nStep 3: Running Validation Analysis...\n")
        self.execute_python("scripts/detection/test.py")
    
    def run_merge_results(self):
        """Step 4: Merge detection and OCR results"""
        print("\nStep 4: Merging Detection and OCR Results...\n")
        self.execute_python("scripts/utils/merge_detection_ocr.py")
    
    def generate_summary(self):
        """Generate summary statistics from complete results"""
        print("\nGenerating Summary Statistics...\n")
        
        complete_results_dir = self.output_dir / 'complete_results'
        if not complete_results_dir.exists():
            print("Warning: No complete results directory found")
            return None
        
        # Load all complete result files
        result_files = list(complete_results_dir.glob('*_detections.json'))
        
        if not result_files:
            print("Warning: No result files found")
            return None
        
        all_results = []
        for result_file in result_files:
            with open(result_file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        
        # Calculate statistics
        stats = {
            'total_images': len(all_results),
            'total_components': 0,
            'components_with_ocr': 0,
            'average_confidence': 0,
            'class_distribution': {},
            'images': []
        }
        
        total_confidence = 0
        total_component_count = 0
        
        for img_result in all_results:
            img_stats = {
                'name': img_result.get('image', 'unknown'),
                'components': len(img_result.get('components', [])),
                'components_with_text': 0
            }
            
            for comp in img_result.get('components', []):
                total_component_count += 1
                stats['total_components'] += 1
                total_confidence += comp.get('confidence', 0)
                
                # Count class distribution
                label = comp.get('class', 'Component')
                stats['class_distribution'][label] = stats['class_distribution'].get(label, 0) + 1
                
                # Count OCR results
                if comp.get('ocr_texts') and len(comp.get('ocr_texts', [])) > 0:
                    stats['components_with_ocr'] += 1
                    img_stats['components_with_text'] += 1
            
            stats['images'].append(img_stats)
        
        stats['average_confidence'] = total_confidence / total_component_count if total_component_count > 0 else 0
        
        # Save summary
        summary_path = self.results_dir / 'summary_statistics.json'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved summary statistics to: {summary_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total Images Processed: {stats['total_images']}")
        print(f"Total Components Detected: {stats['total_components']}")
        
        ocr_percentage = (stats['components_with_ocr'] / stats['total_components'] * 100) if stats['total_components'] > 0 else 0
        print(f"Components with OCR Text: {stats['components_with_ocr']} ({ocr_percentage:.1f}%)")
        print(f"Average Confidence: {stats['average_confidence']*100:.2f}%")
        
        print("\nClass Distribution:")
        for class_name, count in sorted(stats['class_distribution'].items()):
            print(f"  {class_name}: {count}")
        print(f"{'='*80}\n")
        
        return stats
    
    def run(self):
        """Run the complete pipeline"""
        print(f"\n{'='*80}")
        print("AIRCRAFT COMPONENT DETECTION PIPELINE")
        print("Complete Detection, Classification, and OCR")
        print(f"{'='*80}\n")
        
        try:
            # Step 1: YOLO Detection
            self.run_yolo_detection()
            
            # Step 2: YOLO with OCR
            self.run_yolo_with_ocr()
            
            # Step 3: Merge results
            self.run_merge_results()
            
            # Step 4: Generate summary
            self.generate_summary()
            
            print("\nPipeline completed successfully!\n")
            print(f"Results available in: {self.results_dir}")
            print("\nGenerated files:")
            print("  - Individual detection JSON files")
            print("  - Merged detection + OCR results")
            print("  - summary_statistics.json (pipeline statistics)")
            print()
            
        except Exception as error:
            print(f"\nPipeline failed: {error}")
            sys.exit(1)


if __name__ == '__main__':
    pipeline = ComponentDetectionPipeline()
    pipeline.run()

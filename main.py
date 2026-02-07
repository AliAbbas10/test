"""Aircraft Component Detection Training Pipeline

Complete end-to-end training pipeline orchestrator that runs:
    1. Grid search for hyperparameter optimization (144 combinations × 50 epochs)
    2. Analysis to identify best configuration
    3. Full model training with optimal hyperparameters (150 epochs)
    4. Model testing and metrics evaluation
    5. JSON detection generation
    6. OCR text extraction
    7. Merge detection + OCR results

Usage:
    python main.py                    # Run full pipeline (training + inference)
    python main.py --skip-grid-search # Skip grid search, use existing results
    python main.py --grid-search-only # Only run grid search and analysis
    python main.py --train-only       # Only run training (no inference)
    python main.py --inference-only   # Only run testing + detection + OCR
"""

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime


def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return success status
    
    Args:
        script_path: Path to the Python script to run
        description: Human-readable description for logging
        
    Returns:
        True if script completed successfully, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"STEP: {description}")
    print("=" * 80)
    print(f"[INFO] Running: {script_path}")
    print(f"[INFO] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=Path.cwd(),
            text=True
        )
        print(f"\n[SUCCESS] {description} completed successfully")
        print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed with exit code {e.returncode}")
        print(f"[INFO] Failed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return False
    except KeyboardInterrupt:
        print(f"\n[WARNING] {description} interrupted by user")
        return False


def check_grid_search_results() -> bool:
    """Check if grid search results already exist"""
    results_file = Path('grid_search_results.csv')
    return results_file.exists()


def main():
    parser = argparse.ArgumentParser(
        description='Aircraft Component Detection Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline (all steps)
  python main.py --skip-grid-search # Use existing grid search results
  python main.py --grid-search-only # Only optimize hyperparameters
  python main.py --train-only       # Only training (no inference)
  python main.py --inference-only   # Only testing/detection/OCR
  python main.py --skip-ocr         # Skip OCR step (faster inference)
        """
    )
    
    parser.add_argument(
        '--skip-grid-search',
        action='store_true',
        help='Skip grid search and use existing results (if available)'
    )
    parser.add_argument(
        '--grid-search-only',
        action='store_true',
        help='Only run grid search and analysis, skip training and inference'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only run training, skip grid search and inference'
    )
    parser.add_argument(
        '--inference-only',
        action='store_true',
        help='Only run inference (testing, detection, OCR), skip training'
    )
    parser.add_argument(
        '--skip-ocr',
        action='store_true',
        help='Skip OCR step in inference pipeline (faster)'
    )
    
    args = parser.parse_args()
    
    # Validate script paths
    grid_search_script = Path('scripts/training/grid_search.py')
    analysis_script = Path('scripts/analysis/analyze_grid_search.py')
    train_script = Path('scripts/training/train.py')
    test_script = Path('scripts/detection/test.py')
    detect_json_script = Path('scripts/detection/detect_json.py')
    ocr_script = Path('scripts/detection/ocr_detect.py')
    merge_script = Path('scripts/utils/merge_detection_ocr.py')
    
    missing_scripts = []
    for script in [grid_search_script, analysis_script, train_script, test_script, detect_json_script, ocr_script, merge_script]:
        if not script.exists():
            missing_scripts.append(str(script))
    
    if missing_scripts:
        print("[ERROR] Missing required scripts:")
        for script in missing_scripts:
            print(f"  - {script}")
        sys.exit(1)
    
    # Print pipeline configuration
    print("\n" + "=" * 80)
    print("AIRCRAFT COMPONENT DETECTION TRAINING PIPELINE")
    print("=" * 80)
    print(f"[INFO] Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine execution mode
    if args.inference_only:
        print("\n[MODE] Inference only (testing, detection, OCR)")
        run_grid_search = False
        run_analysis = False
        run_training = False
        run_testing = True
        run_detection = True
        run_ocr = not args.skip_ocr
        run_merge = not args.skip_ocr
    elif args.train_only:
        print("\n[MODE] Training only (using existing hyperparameters)")
        run_grid_search = False
        run_analysis = False
        run_training = True
        run_testing = False
        run_detection = False
        run_ocr = False
        run_merge = False
    elif args.grid_search_only:
        print("\n[MODE] Grid search and analysis only")
        run_grid_search = True
        run_analysis = True
        run_training = False
        run_testing = False
        run_detection = False
        run_ocr = False
        run_merge = False
    elif args.skip_grid_search:
        print("\n[MODE] Skip grid search (using existing results)")
        run_grid_search = False
        run_analysis = False
        run_training = True
        run_testing = True
        run_detection = True
        run_ocr = not args.skip_ocr
        run_merge = not args.skip_ocr
        
        if not check_grid_search_results():
            print("[WARNING] No existing grid search results found!")
            print("[INFO] Training will use default hyperparameters")
    else:
        print("\n[MODE] Full pipeline (training + inference)")
        run_grid_search = True
        run_analysis = True
        run_training = True
        run_testing = True
        run_detection = True
        run_ocr = not args.skip_ocr
        run_merge = not args.skip_ocr
    
    print("\n[PIPELINE] Steps to execute:")
    step_num = 1
    if run_grid_search:
        print(f"  {step_num}. Grid search hyperparameter optimization")
        step_num += 1
        print(f"  {step_num}. Analyze results and identify best configuration")
        step_num += 1
    if run_training:
        print(f"  {step_num}. Full model training with optimal hyperparameters")
        step_num += 1
    if run_testing:
        print(f"  {step_num}. Model testing and metrics evaluation")
        step_num += 1
    if run_detection:
        print(f"  {step_num}. Generate detection JSON outputs")
        step_num += 1
    if run_ocr:
        print(f"  {step_num}. OCR text extraction from detections")
        step_num += 1
    if run_merge:
        print(f"  {step_num}. Merge detection and OCR results")
        step_num += 1
    print("=" * 80)
    
    # Execute pipeline steps
    success = True
    
    # Step 1: Grid Search (if enabled)
    if run_grid_search:
        success = run_script(
            str(grid_search_script),
            "Grid Search Hyperparameter Optimization"
        )
        if not success:
            print("\n[ERROR] Pipeline failed at grid search step")
            sys.exit(1)
    
    # Step 2: Analysis (if enabled)
    if run_analysis:
        success = run_script(
            str(analysis_script),
            "Grid Search Results Analysis"
        )
        if not success:
            print("\n[ERROR] Pipeline failed at analysis step")
            sys.exit(1)
    
    # Step 3: Training (if enabled)
    if run_training:
        success = run_script(
            str(train_script),
            "Full Model Training (150 epochs)"
        )
        if not success:
            print("\n[ERROR] Pipeline failed at training step")
            sys.exit(1)
    
    # Step 4: Testing (if enabled)
    if run_testing:
        success = run_script(
            str(test_script),
            "Model Testing and Metrics Evaluation"
        )
        if not success:
            print("\n[WARNING] Testing step failed, continuing with pipeline...")
    
    # Step 5: Detection JSON Generation (if enabled)
    if run_detection:
        success = run_script(
            str(detect_json_script),
            "Generate Detection JSON Outputs"
        )
        if not success:
            print("\n[WARNING] Detection JSON generation failed, continuing...")
    
    # Step 6: OCR Detection (if enabled)
    if run_ocr:
        success = run_script(
            str(ocr_script),
            "OCR Text Extraction from Detections"
        )
        if not success:
            print("\n[WARNING] OCR step failed, continuing...")
    
    # Step 7: Merge Detection + OCR (if enabled)
    if run_merge:
        success = run_script(
            str(merge_script),
            "Merge Detection and OCR Results"
        )
        if not success:
            print("\n[WARNING] Merge step failed")
    
    # Pipeline completion
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"[INFO] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if run_training:
        print("[TRAINING RESULTS]")
        print("  - Final trained model (yolov8l): runs/detect/training/weights/best.pt")
        print("  - Validation images: runs/detect/training/validation_results/")
    
    if run_testing:
        print("\n[TESTING RESULTS]")
        print("  - Test metrics: runs/detect/test/")
        print("  - Annotated test images with metrics")
    
    if run_detection:
        print("\n[DETECTION RESULTS]")
        print("  - Detection JSONs: demo/all_detections/")
    
    if run_ocr:
        print("\n[OCR RESULTS]")
        print("  - OCR JSONs: demo/all_ocr_results/")
    
    if run_merge:
        print("\n[FINAL RESULTS]")
        print("  - Complete results (detection + OCR): demo/complete_results/")
    
    if run_analysis and not run_training:
        print("\n[GRID SEARCH RESULTS]")
        print("  - Best hyperparameters: grid_search_results.csv")
        print("  - Best grid search model (yolov8s): models/best_trained_model.pt")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()


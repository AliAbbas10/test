from ultralytics import YOLO
from pathlib import Path
import torch
import pandas as pd
from itertools import product
import json
from datetime import datetime

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 0
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = 'cpu'
        print("No GPU detected, using CPU")

    # Define hyperparameter grid
    param_grid = {
        'lr0': [0.0005, 0.001, 0.002],           # Initial learning rate
        'weight_decay': [0.0001, 0.0005, 0.001], # Weight decay
        'hsv_h': [0.01, 0.02],                   # Hue augmentation
        'hsv_s': [0.7, 0.75],                    # Saturation augmentation
        'degrees': [10, 15],                     # Rotation degrees
        'mosaic': [0.8, 1.0],                    # Mosaic augmentation
    }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH: Testing {len(combinations)} hyperparameter combinations")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        
        print(f"\n{'='*80}")
        print(f"Experiment {idx}/{len(combinations)}")
        print(f"{'='*80}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        print(f"{'='*80}\n")
        
        # Create model
        model = YOLO('models/yolov8s.pt')
        
        # Train with current hyperparameters
        try:
            train_results = model.train(
                data='config/train.yaml',
                epochs=50,                      # Shorter for grid search
                imgsz=1024,
                batch=4,
                patience=15,
                device=device,
                project='grid_search',
                name=f'exp_{idx}',
                verbose=False,                  # Less verbose output
                plots=False,                    # Skip plots for speed
                workers=0,
                
                # Grid search parameters
                lr0=params['lr0'],
                weight_decay=params['weight_decay'],
                hsv_h=params['hsv_h'],
                hsv_s=params['hsv_s'],
                degrees=params['degrees'],
                mosaic=params['mosaic'],
                
                # Fixed parameters
                lrf=0.001,
                hsv_v=0.45,
                translate=0.15,
                scale=0.7,
                shear=8,
                perspective=0.001,
                flipud=0.5,
                fliplr=0.5,
                mixup=0.15,
                copy_paste=0.1,
                erasing=0.5,
            )
            
            # Load results CSV to get metrics
            results_csv = Path(f'grid_search/exp_{idx}/results.csv')
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                
                # Get best metrics
                best_map50 = df['metrics/mAP50(B)'].max()
                best_map = df['metrics/mAP50-95(B)'].max()
                final_precision = df['metrics/precision(B)'].iloc[-1]
                final_recall = df['metrics/recall(B)'].iloc[-1]
                
                result = {
                    'experiment': idx,
                    **params,
                    'best_mAP50': best_map50,
                    'best_mAP50-95': best_map,
                    'final_precision': final_precision,
                    'final_recall': final_recall,
                }
                
                results.append(result)
                
                print(f"\nExperiment {idx} completed:")
                print(f"   mAP@50:    {best_map50:.4f}")
                print(f"   mAP@50-95: {best_map:.4f}")
                print(f"   Precision: {final_precision:.4f}")
                print(f"   Recall:    {final_recall:.4f}")
            else:
                print(f"Results file not found for experiment {idx}")
                
        except Exception as e:
            print(f"Error in experiment {idx}: {e}")
            continue
    
    # Save all results
    print(f"\n{'='*80}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*80}\n")
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'grid_search_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}\n")
        
        # Sort by best mAP@50
        results_df_sorted = results_df.sort_values('best_mAP50', ascending=False)
        
        print("\n" + "="*80)
        print("TOP 5 CONFIGURATIONS (by mAP@50)")
        print("="*80)
        print(results_df_sorted.head(5).to_string(index=False))
        
        print("\n" + "="*80)
        print("BEST CONFIGURATION")
        print("="*80)
        best = results_df_sorted.iloc[0]
        print(f"\nExperiment: {int(best['experiment'])}")
        print(f"Location: grid_search/exp_{int(best['experiment'])}/weights/best.pt")
        print(f"\nHyperparameters:")
        for param in param_names:
            print(f"  {param:15s}: {best[param]}")
        print(f"\nMetrics:")
        print(f"  mAP@50:        {best['best_mAP50']:.4f}")
        print(f"  mAP@50-95:     {best['best_mAP50-95']:.4f}")
        print(f"  Precision:     {best['final_precision']:.4f}")
        print(f"  Recall:        {best['final_recall']:.4f}")
        
        print("\n" + "="*80)
        print("PARAMETER ANALYSIS")
        print("="*80)
        
        # Analyze impact of each parameter
        for param in param_names:
            print(f"\n{param}:")
            grouped = results_df.groupby(param)['best_mAP50'].agg(['mean', 'std', 'count'])
            print(grouped.to_string())
        
    else:
        print("No results collected.")
    
    print("\n" + "="*80)

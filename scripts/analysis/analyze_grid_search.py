import pandas as pd
from pathlib import Path
import shutil

results = []
grid_search_dir = Path('runs/detect/grid_search')

print("Analyzing grid search results...")
for exp_dir in sorted(grid_search_dir.glob('exp_*'), key=lambda x: int(x.name.replace('exp_', ''))):
    results_csv = exp_dir / 'results.csv'
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            exp_num = exp_dir.name.replace('exp_', '')
            best_map50 = df['metrics/mAP50(B)'].max()
            best_map = df['metrics/mAP50-95(B)'].max()
            final_precision = df['metrics/precision(B)'].iloc[-1]
            final_recall = df['metrics/recall(B)'].iloc[-1]
            
            results.append({
                'experiment': exp_num,
                'best_mAP50': best_map50,
                'best_mAP50-95': best_map,
                'final_precision': final_precision,
                'final_recall': final_recall
            })
        except Exception as e:
            print(f"Error processing {exp_dir.name}: {e}")

results_df = pd.DataFrame(results)

# Calculate F1 score
results_df['f1_score'] = 2 * (results_df['final_precision'] * results_df['final_recall']) / (results_df['final_precision'] + results_df['final_recall'])

results_df.to_csv('grid_search_results.csv', index=False)
print(f"\nResults saved to grid_search_results.csv")

results_df = results_df.sort_values('f1_score', ascending=False)
print('\n' + '='*80)
print('TOP 10 EXPERIMENTS BY F1 SCORE:')
print('='*80)
print(results_df.head(10).to_string(index=False))

print('\n' + '='*80)
print('BEST EXPERIMENT (BY F1 SCORE):')
print('='*80)
best = results_df.iloc[0]
print(f"\nExperiment: exp_{best['experiment']}")
print(f"Location: runs/detect/grid_search/exp_{best['experiment']}/weights/best.pt")
print(f"\nMetrics:")
print(f"  F1 Score:      {best['f1_score']:.4f}")
print(f"  Precision:     {best['final_precision']:.4f}")
print(f"  Recall:        {best['final_recall']:.4f}")
print(f"  mAP@50:        {best['best_mAP50']:.4f}")
print(f"  mAP@50-95:     {best['best_mAP50-95']:.4f}")

# Copy best model to standardized location
best_model_source = grid_search_dir / f"exp_{best['experiment']}" / 'weights' / 'best.pt'
best_model_dest = Path('models/best_trained_model.pt')

if best_model_source.exists():
    best_model_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model_source, best_model_dest)
    print(f"\n[INFO] Best model copied to: {best_model_dest}")
    print(f"   Use this for testing and inference!")
else:
    print(f"\n[WARNING] Best model file not found at {best_model_source}")

print('='*80)

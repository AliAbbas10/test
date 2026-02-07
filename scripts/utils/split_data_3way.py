#!/usr/bin/env python3
"""
Split YOLO dataset into train/validation/test sets (3-way split)

Ensures complete separation between train, validation, and test sets
to prevent data leakage and get accurate performance metrics.

Usage:
    python scripts/utils/split_data_3way.py <input_dir> --train 70 --val 15 --test 15
    
Example:
    python scripts/utils/split_data_3way.py data/raw_images --train 70 --val 15 --test 15
"""

import argparse
import random
import shutil
from pathlib import Path


def split_yolo_dataset_3way(
    input_dir: Path,
    output_root: Path,
    train_percent: int,
    val_percent: int,
    test_percent: int,
    seed: int = 42
) -> None:
    """Split dataset into train/validation/test sets"""
    
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if train_percent + val_percent + test_percent != 100:
        raise ValueError(f"Percentages must sum to 100, got {train_percent + val_percent + test_percent}")
    
    print(f"\n{'='*80}")
    print(f"3-WAY DATASET SPLIT")
    print(f"{'='*80}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_root}")
    print(f"Split: {train_percent}% train, {val_percent}% val, {test_percent}% test")
    print(f"{'='*80}\n")
    
    # Output directories
    train_images_dir = output_root / "images" / "train"
    val_images_dir = output_root / "images" / "validation"
    test_images_dir = output_root / "images" / "test"
    train_labels_dir = output_root / "labels" / "train"
    val_labels_dir = output_root / "labels" / "validation"
    test_labels_dir = output_root / "labels" / "test"
    
    for d in [train_images_dir, val_images_dir, test_images_dir,
              train_labels_dir, val_labels_dir, test_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Find all .png files
    png_files = sorted(input_dir.glob("*.png"))
    
    # Build paired samples: (png_path, txt_path)
    samples = []
    for png_path in png_files:
        base = png_path.stem
        txt_path = input_dir / f"{base}.txt"
        
        # Must have matching annotation .txt
        if not txt_path.exists():
            print(f"Warning: Skipping {png_path.name} - no label file found")
            continue
        
        # Omit classes.txt specifically
        if txt_path.name == "classes.txt":
            continue
        
        samples.append((png_path, txt_path))
    
    if not samples:
        raise RuntimeError("No valid (png, txt) pairs found in the input directory.")
    
    print(f"Found {len(samples)} image-label pairs\n")
    
    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(samples)
    
    # Calculate split indices
    n = len(samples)
    n_train = int(n * train_percent / 100)
    n_val = int(n * val_percent / 100)
    # Rest goes to test to ensure all samples are used
    n_test = n - n_train - n_val
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    print(f"Split breakdown:")
    print(f"  Train:      {len(train_samples):3d} samples ({len(train_samples)/n*100:.1f}%)")
    print(f"  Validation: {len(val_samples):3d} samples ({len(val_samples)/n*100:.1f}%)")
    print(f"  Test:       {len(test_samples):3d} samples ({len(test_samples)/n*100:.1f}%)")
    print(f"  Total:      {n:3d} samples\n")
    
    # Copy files
    def copy_split(samples, img_dir, lbl_dir, split_name):
        print(f"Copying {split_name} set...")
        for png_path, txt_path in samples:
            # Copy image
            shutil.copy2(png_path, img_dir / png_path.name)
            # Copy label
            shutil.copy2(txt_path, lbl_dir / txt_path.name)
        print(f"  ✓ {len(samples)} files copied to {split_name}\n")
    
    copy_split(train_samples, train_images_dir, train_labels_dir, "train")
    copy_split(val_samples, val_images_dir, val_labels_dir, "validation")
    copy_split(test_samples, test_images_dir, test_labels_dir, "test")
    
    print(f"{'='*80}")
    print("Dataset split complete!")
    print(f"{'='*80}\n")
    print("Directory structure:")
    print(f"  {output_root}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/         ({len(train_samples)} images)")
    print(f"    │   ├── validation/    ({len(val_samples)} images)")
    print(f"    │   └── test/          ({len(test_samples)} images)")
    print(f"    └── labels/")
    print(f"        ├── train/         ({len(train_samples)} labels)")
    print(f"        ├── validation/    ({len(val_samples)} labels)")
    print(f"        └── test/          ({len(test_samples)} labels)\n")
    
    print("IMPORTANT: Update config/train.yaml to add test path:")
    print("test: images/test\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train/validation/test sets"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing .png images and .txt labels"
    )
    parser.add_argument(
        "--train",
        type=int,
        default=70,
        help="Training set percentage (default: 70)"
    )
    parser.add_argument(
        "--val",
        type=int,
        default=15,
        help="Validation set percentage (default: 15)"
    )
    parser.add_argument(
        "--test",
        type=int,
        default=15,
        help="Test set percentage (default: 15)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/datasets/aircraft-components",
        help="Output directory for split dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_root = Path(args.output)
    
    split_yolo_dataset_3way(
        input_dir=input_dir,
        output_root=output_root,
        train_percent=args.train,
        val_percent=args.val,
        test_percent=args.test,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

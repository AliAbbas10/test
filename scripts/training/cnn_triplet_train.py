"""
Triplet CNN Training for Aircraft Component Classification

This script trains a triplet network to learn embeddings for component classification.
Works in conjunction with YOLO detections to improve classification accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
import cv2
import numpy as np
import random
from collections import defaultdict
import json


class TripletComponentDataset(Dataset):
    """Dataset for generating triplet samples from labeled component images"""
    
    def __init__(self, data_dir, labels_dir, classes_file, transform=None):
        self.data_dir = Path(data_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        
        # Load class names
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines() if line.strip()]
        
        # Build dataset: group components by class
        self.components_by_class = defaultdict(list)
        self._build_dataset()
        
        # Filter out classes with less than 2 examples
        self.valid_classes = [cls for cls, items in self.components_by_class.items() 
                             if len(items) >= 2]
        
        print(f"\nDataset Statistics:")
        print(f"Total classes: {len(self.classes)}")
        print(f"Valid classes (>=2 examples): {len(self.valid_classes)}")
        for cls in self.valid_classes:
            class_name = self.classes[cls] if cls < len(self.classes) else f"Unknown({cls})"
            print(f"  {class_name}: {len(self.components_by_class[cls])} examples")
    
    def _build_dataset(self):
        """Extract individual components from images using label files"""
        image_files = list(self.data_dir.glob('*.png')) + list(self.data_dir.glob('*.jpg'))
        
        for img_path in image_files:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Load corresponding label file
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Parse labels and extract components
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Extract component crop
                    component_crop = image[y1:y2, x1:x2]
                    
                    if component_crop.size == 0:
                        continue
                    
                    self.components_by_class[class_id].append({
                        'image': component_crop,
                        'class_id': class_id,
                        'source_image': img_path.name
                    })
    
    def __len__(self):
        # Each epoch will generate triplets from all valid classes
        return len(self.valid_classes) * 50  # 50 triplets per class
    
    def __getitem__(self, idx):
        """Generate a triplet: (anchor, positive, negative)"""
        
        # Select anchor class
        anchor_class = random.choice(self.valid_classes)
        
        # Select anchor and positive from same class
        anchor, positive = random.sample(self.components_by_class[anchor_class], 2)
        
        # Select negative from different class
        negative_class = random.choice([c for c in self.valid_classes if c != anchor_class])
        negative = random.choice(self.components_by_class[negative_class])
        
        # Process images
        anchor_img = self._process_image(anchor['image'])
        positive_img = self._process_image(positive['image'])
        negative_img = self._process_image(negative['image'])
        
        return anchor_img, positive_img, negative_img, anchor_class
    
    def _process_image(self, img):
        """Resize and normalize image"""
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to fixed size
        img = cv2.resize(img, (128, 128))
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)
        
        return img


class EmbeddingNetwork(nn.Module):
    """CNN for learning component embeddings"""
    
    def __init__(self, embedding_dim=128):
        super(EmbeddingNetwork, self).__init__()
        
        # Use ResNet18 as backbone
        resnet = models.resnet18(pretrained=True)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        # L2 normalize embeddings
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class TripletLoss(nn.Module):
    """Triplet loss with margin"""
    
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # Calculate distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Triplet loss
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def train_triplet_network():
    """Main training function"""
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("No GPU detected, using CPU")
    
    # Paths
    train_images = Path('./data/training/datasets/aircraft-components/images/train')
    train_labels = Path('./data/training/datasets/aircraft-components/labels/train')
    classes_file = Path('./data/training/datasets/aircraft-components/classes.txt')
    
    # Check paths exist
    if not train_images.exists():
        print(f"Error: Training images not found at {train_images}")
        return
    
    # Create dataset
    print("\nBuilding triplet dataset...")
    dataset = TripletComponentDataset(
        data_dir=train_images,
        labels_dir=train_labels,
        classes_file=classes_file
    )
    
    if len(dataset.valid_classes) < 2:
        print("Error: Need at least 2 classes with 2+ examples each")
        return
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Create model
    print("\nInitializing embedding network...")
    model = EmbeddingNetwork(embedding_dim=128).to(device)
    
    # Loss and optimizer
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training
    num_epochs = 50
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*80)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (anchor, positive, negative, labels) in enumerate(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            
            # Calculate loss
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path('models/triplet_embedding_best.pt')
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'classes': dataset.classes
            }, save_path)
            print(f"  Saved best model (loss: {avg_loss:.4f})")
    
    # Save final model
    final_path = Path('models/triplet_embedding_final.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'classes': dataset.classes
    }, final_path)
    
    print("\n" + "="*80)
    print(f"Training complete!")
    print(f"Best model saved to: models/triplet_embedding_best.pt")
    print(f"Final model saved to: models/triplet_embedding_final.pt")
    print(f"Best loss: {best_loss:.4f}")
    print("="*80)


if __name__ == '__main__':
    train_triplet_network()
    
# 1. Data Preparation
#    ├── Load aircraft images from training directory
#    ├── Read YOLO label files (.txt format)
#    ├── Extract component crops using bounding box coordinates  
#    └── Group components by class ID

# 2. Triplet Generation (during training)
#    ├── Select random anchor class
#    ├── Pick anchor + positive from same class
#    ├── Pick negative from different class
#    └── Resize all to 128x128, normalize to [0,1]

# 3. Network Architecture
#    ├── ResNet-18 backbone (feature extraction)
#    ├── Global Average Pooling
#    ├── FC Layer: 512 → 256 (ReLU, Dropout)
#    ├── FC Layer: 256 → 128 (embedding output)
#    └── L2 Normalization

# 4. Training Process
#    ├── Forward pass: Generate embeddings for triplet
#    ├── Calculate triplet loss
#    ├── Backpropagation and optimization (Adam)
#    ├── Learning rate scheduling (StepLR)
#    └── Save best model based on lowest loss

# 5. Output
#    ├── Best model: triplet_embedding_best.pt
#    └── Final model: triplet_embedding_final.pt
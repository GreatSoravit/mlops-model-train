#!/usr/bin/env python3
"""
Optimized Colon Cancer Cell Classification Training Script
"""

import os
import torch
import torchvision
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime
import json

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import collections
from typing import Dict, Tuple, List
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CCDataset(Dataset):
    """Optimized Dataset class with better error handling and augmentation support"""
    
    def __init__(self, csv_file: str, root_dir: str, transform=None, is_test: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
        if not is_test:
            self.info = pd.read_csv(csv_file)
            self.all_imgs = [str(self.info.iloc[i, 0]) + '.png' for i in range(len(self.info))]
        else:
            self.all_imgs = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        
        # Class mapping
        self.class_to_idx = {'Cancer': 0, 'Connective': 1, 'Immune': 2, 'Normal': 3}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.all_imgs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.is_test:
            img_name = self.all_imgs[idx]
            img_loc = os.path.join(self.root_dir, img_name)
            image_id = img_name[:-4]  # Remove .png extension
            
            image = io.imread(img_loc)
            if self.transform:
                image = self.transform(image)
            else:
                image = torchvision.transforms.functional.to_tensor(image)
                
            return {'image': image, 'id': image_id}
        else:
            img_loc = os.path.join(self.root_dir, self.all_imgs[idx])
            image = io.imread(img_loc)
            
            if self.transform:
                image = self.transform(image)
            else:
                image = torchvision.transforms.functional.to_tensor(image)

            celltype = self.info.iloc[idx, 1]
            class_no = self.class_to_idx[celltype]
            
            return {
                'image': image, 
                'celltype': celltype, 
                'class': class_no,
                'id': self.info.iloc[idx, 0]
            }

class ImprovedCNN(nn.Module):
    """Improved CNN architecture with batch normalization and dropout"""
    
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.5):
        super(ImprovedCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

def get_transforms(mean: List[float], std: List[float], is_training: bool = True):
    """Get data transforms with augmentation for training"""
    
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),  # Slightly larger than original
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

def compute_dataset_stats(dataset) -> Tuple[List[float], List[float]]:
    """Compute mean and std for dataset normalization"""
    logger.info("Computing dataset statistics...")
    
    imgs_stack = torch.stack([sample['image'] for sample in dataset], dim=3)
    temp_imgs = imgs_stack.view(3, -1)
    imgs_mean = temp_imgs.mean(dim=1).tolist()
    imgs_std = temp_imgs.std(dim=1).tolist()
    
    logger.info(f"Dataset mean: {imgs_mean}")
    logger.info(f"Dataset std: {imgs_std}")
    
    return imgs_mean, imgs_std

def get_class_distribution(dataset) -> Dict[str, int]:
    """Get class distribution from dataset"""
    class_counts = {'Cancer': 0, 'Connective': 0, 'Immune': 0, 'Normal': 0}
    
    for sample in dataset:
        if 'celltype' in sample:
            class_counts[sample['celltype']] += 1
    
    return class_counts

def create_weighted_sampler(dataset):
    """Create weighted sampler for class balancing"""
    class_counts = get_class_distribution(dataset)
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    # Create sample weights
    sample_weights = []
    for sample in dataset:
        if 'celltype' in sample:
            sample_weights.append(class_weights[sample['celltype']])
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def create_data_loaders(train_csv: str, train_dir: str, batch_size: int = 32, 
                       val_split: float = 0.2, num_workers: int = 8):
    """Create optimized data loaders with proper validation split"""
    
    # First pass to compute statistics
    temp_dataset = CCDataset(train_csv, train_dir)
    mean, std = compute_dataset_stats(temp_dataset)
    
    # Create datasets with transforms
    train_transform = get_transforms(mean, std, is_training=True)
    val_transform = get_transforms(mean, std, is_training=False)
    
    full_dataset = CCDataset(train_csv, train_dir, transform=train_transform)
    val_dataset = CCDataset(train_csv, train_dir, transform=val_transform)
    
    # Create weighted sampler and split
    weighted_sampler = create_weighted_sampler(full_dataset)
    dataset_size = len(weighted_sampler)
    sampler_indices = list(weighted_sampler)
    
    val_split_index = int(np.floor(val_split * dataset_size))
    train_idx = sampler_indices[val_split_index:]
    val_idx = sampler_indices[:val_split_index]
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, mean, std

def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, data in enumerate(train_loader):
        images = data['image'].to(device, non_blocking=True)
        labels = data['class'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler and isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in val_loader:
            images = data['image'].to(device, non_blocking=True)
            labels = data['class'].to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_model(config: Dict):
    """Main training function"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Create data loaders
    train_loader, val_loader, mean, std = create_data_loaders(
        train_csv=config['train_csv'],
        train_dir=config['train_dir'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        num_workers=config['num_workers']
    )
    
    # Log class distributions
    logger.info(f"Training set distribution: {get_class_distribution(train_loader.dataset)}")
    logger.info(f"Validation set distribution: {get_class_distribution(val_loader.dataset)}")
    
    # Create model
    model = ImprovedCNN(
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function with class weights
    class_counts = get_class_distribution(train_loader.dataset)
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        total_samples / class_counts['Cancer'],
        total_samples / class_counts['Connective'], 
        total_samples / class_counts['Immune'],
        total_samples / class_counts['Normal']
    ]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    
    # Scheduler
    if config['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    elif config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        scheduler = None
    
    # Training loop
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    logger.info("Starting training...")
    
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            elif not isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'mean': mean,
                'std': std,
                'class_to_idx': train_loader.dataset.class_to_idx
            }, config['model_path'])
            logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Log progress
        if epoch % config['log_interval'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch+1}/{config['epochs']}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"LR: {current_lr:.6f}"
            )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
    
    with open(config['history_path'], 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history

def plot_training_history(history_path: str):
    """Plot training history"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Colon Cancer Cell Classifier')
    parser.add_argument('--train_csv', type=str, default='data/train.csv', help='Training CSV file')
    parser.add_argument('--train_dir', type=str, default='data/train/train/', help='Training images directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'plateau'], help='LR scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--step_size', type=float, default=15, help='Step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_path', type=str, default='best_colon_cancer_model.pth', help='Model save path')
    
    parser.add_argument('--config', type=str, default=None, help="Optional path to JSON config file")
     
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # Load config file if provided
    if args.config and os.path.isfile(args.config):
        with open(args.config, 'r') as f:
            json_data = json.load(f)
            if "best_params" in json_data:
                json_data = json_data["best_params"]

        # Override defaults from JSON
        for key, value in json_data.items():
            if hasattr(args, key):
                setattr(args, key, value)             
    
    config = {
        'train_csv': args.train_csv,
        'train_dir': args.train_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'weight_decay': args.weight_decay,
        'dropout_rate': args.dropout_rate,
        'momentum': args.momentum,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'seed': args.seed,
        'num_classes': 4,
        'val_split': 0.2,
        'num_workers': 8,
        'log_interval': 5,
        'model_path': args.model_path,
        'history_path': 'training_history.json'
    }
    
    logger.info("Starting training with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Train model
    model, history = train_model(config)
    
    # Plot results
    plot_training_history(config['history_path'])
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
"""
Main execution script for Sunflower vs Rose Classification
This script orchestrates the entire pipeline: data preparation, training, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path

# Import custom modules
from data_loader import DataLoader as FlowerDataLoader, create_sample_dataset
from train import (FlowerDataset, FlowerClassifier, SimpleFlowerClassifier, 
                   Trainer, get_transforms, plot_confusion_matrix)
from inference import FlowerPredictor, predict_directory

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sunflower vs Rose Classification')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing dataset')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset for testing')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of sample images per class')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size for training')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='resnet',
                       choices=['resnet', 'simple'],
                       help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights for ResNet')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Mode arguments
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'predict'],
                       help='Mode: train, eval, or predict')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                       help='Path to save/load model')
    parser.add_argument('--predict_path', type=str, default=None,
                       help='Path to image or directory for prediction')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(args):
    """Train the classification model"""
    
    print("\n" + "="*80)
    print("SUNFLOWER vs ROSE IMAGE CLASSIFICATION - TRAINING")
    print("="*80 + "\n")
    
    # Setup data
    data_loader = FlowerDataLoader(data_dir=args.data_dir)
    
    if args.create_sample:
        print("Creating sample dataset...")
        data_loader.setup_directories()
        create_sample_dataset(data_dir=args.data_dir, num_samples=args.num_samples)
        data_loader.prepare_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Get data paths
    train_paths, train_labels = data_loader.get_data_paths('train')
    val_paths, val_labels = data_loader.get_data_paths('val')
    
    if len(train_paths) == 0:
        print("\n⚠ ERROR: No training data found!")
        print("Please either:")
        print("  1. Use --create_sample flag to create sample data")
        print("  2. Add your images to data/raw/sunflowers and data/raw/roses")
        print("     Then run data_loader.py to prepare the dataset")
        return
    
    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(image_size=args.image_size)
    
    # Create datasets
    train_dataset = FlowerDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = FlowerDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    if args.model_type == 'resnet':
        model = FlowerClassifier(pretrained=args.pretrained)
        print("Model: ResNet18 (Transfer Learning)")
    else:
        model = SimpleFlowerClassifier()
        print("Model: Simple CNN (From Scratch)")
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Create trainer
    trainer = Trainer(model, device, train_loader, val_loader, 
                     criterion, optimizer, scheduler)
    
    # Train model
    best_val_acc = trainer.fit(num_epochs=args.num_epochs, save_path=args.model_path)
    
    # Plot training metrics
    trainer.plot_metrics(save_path='training_metrics.png')
    
    # Evaluate on validation set with best model
    print("\nEvaluating best model on validation set...")
    
    # Load best model
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get predictions
    _, _, all_preds, all_labels = trainer.validate()
    
    # Plot confusion matrix
    class_names = ['Sunflower', 'Rose']
    plot_confusion_matrix(all_labels, all_preds, class_names, 
                         save_path='confusion_matrix.png')
    
    # Print classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nBest model saved to: {args.model_path}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("\nGenerated files:")
    print("  - training_metrics.png")
    print("  - confusion_matrix.png")
    print(f"  - {args.model_path}")

def evaluate_model(args):
    """Evaluate trained model on test set"""
    
    print("\n" + "="*80)
    print("MODEL EVALUATION ON TEST SET")
    print("="*80 + "\n")
    
    # Load data
    data_loader = FlowerDataLoader(data_dir=args.data_dir)
    test_paths, test_labels = data_loader.get_data_paths('test')
    
    if len(test_paths) == 0:
        print("⚠ No test data found!")
        return
    
    print(f"Test samples: {len(test_paths)}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_transform = get_transforms(image_size=args.image_size)
    
    test_dataset = FlowerDataset(test_paths, test_labels, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    # Load model
    if args.model_type == 'resnet':
        model = FlowerClassifier(pretrained=False)
    else:
        model = SimpleFlowerClassifier()
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, device, test_loader, test_loader, criterion, None)
    
    test_loss, test_acc, all_preds, all_labels = trainer.validate()
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Plot confusion matrix
    class_names = ['Sunflower', 'Rose']
    plot_confusion_matrix(all_labels, all_preds, class_names, 
                         save_path='test_confusion_matrix.png')
    
    # Print classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

def predict(args):
    """Make predictions on new images"""
    
    if args.predict_path is None:
        print("⚠ Please provide --predict_path")
        return
    
    print("\n" + "="*80)
    print("MAKING PREDICTIONS")
    print("="*80 + "\n")
    
    predictor = FlowerPredictor(args.model_path)
    
    predict_path = Path(args.predict_path)
    
    if predict_path.is_file():
        # Single image prediction
        predictor.visualize_prediction(str(predict_path), 
                                      save_path='prediction_result.png')
    elif predict_path.is_dir():
        # Directory prediction
        predict_directory(args.model_path, str(predict_path), save_results=True)
    else:
        print(f"⚠ Path not found: {args.predict_path}")

def main():
    """Main execution function"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Execute based on mode
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'eval':
        evaluate_model(args)
    elif args.mode == 'predict':
        predict(args)

if __name__ == "__main__":
    main()

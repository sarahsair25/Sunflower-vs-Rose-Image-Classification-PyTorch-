"""
Utility functions for the flower classification project
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from pathlib import Path
import json

def visualize_augmentations(image_path, transform, num_augmentations=9):
    """
    Visualize different augmentations of an image
    
    Args:
        image_path: Path to the image
        transform: Transformation pipeline
        num_augmentations: Number of augmented versions to show
    """
    image = Image.open(image_path).convert('RGB')
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, ax in enumerate(axes.flat):
        if i == 0:
            ax.imshow(image)
            ax.set_title('Original')
        else:
            # Apply transformation
            augmented = transform(image)
            
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = augmented * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            
            # Convert to numpy and transpose
            img_np = img_denorm.permute(1, 2, 0).numpy()
            
            ax.imshow(img_np)
            ax.set_title(f'Augmentation {i}')
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentations_preview.png', dpi=300, bbox_inches='tight')
    print("✓ Augmentation preview saved to augmentations_preview.png")
    plt.close()

def plot_sample_images(data_dir, num_samples=8):
    """
    Plot sample images from the dataset
    
    Args:
        data_dir: Directory containing the dataset
        num_samples: Number of samples per class to display
    """
    data_path = Path(data_dir)
    train_dir = data_path / 'processed' / 'train'
    
    sunflower_dir = train_dir / 'sunflowers'
    rose_dir = train_dir / 'roses'
    
    # Get sample images
    sunflower_images = list(sunflower_dir.glob('*.jpg'))[:num_samples]
    rose_images = list(rose_dir.glob('*.jpg'))[:num_samples]
    
    # Create plot
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
    
    # Plot sunflowers
    for i, img_path in enumerate(sunflower_images):
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Sunflowers', fontsize=14, fontweight='bold')
    
    # Plot roses
    for i, img_path in enumerate(rose_images):
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Roses', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    print("✓ Sample images saved to sample_images.png")
    plt.close()

def calculate_mean_std(data_dir):
    """
    Calculate mean and standard deviation of the dataset
    Useful for normalization
    
    Args:
        data_dir: Directory containing the dataset
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    data_path = Path(data_dir)
    train_dir = data_path / 'processed' / 'train'
    
    # Get all images
    image_paths = []
    for class_dir in ['sunflowers', 'roses']:
        class_path = train_dir / class_dir
        image_paths.extend(list(class_path.glob('*.jpg')))
        image_paths.extend(list(class_path.glob('*.png')))
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    print(f"Calculating statistics for {len(image_paths)} images...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        
        mean += img_tensor.mean(dim=[1, 2])
        std += img_tensor.std(dim=[1, 2])
    
    mean /= len(image_paths)
    std /= len(image_paths)
    
    print(f"\nDataset Statistics:")
    print(f"Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    
    return mean, std

def analyze_dataset_distribution(data_dir):
    """
    Analyze and visualize dataset distribution
    
    Args:
        data_dir: Directory containing the dataset
    """
    data_path = Path(data_dir)
    processed_dir = data_path / 'processed'
    
    splits = ['train', 'val', 'test']
    classes = ['sunflowers', 'roses']
    
    data = {split: {cls: 0 for cls in classes} for split in splits}
    
    for split in splits:
        split_dir = processed_dir / split
        for cls in classes:
            class_dir = split_dir / cls
            if class_dir.exists():
                count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
                data[split][cls] = count
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Stacked bar chart for splits
    x = np.arange(len(splits))
    width = 0.35
    
    sunflower_counts = [data[split]['sunflowers'] for split in splits]
    rose_counts = [data[split]['roses'] for split in splits]
    
    ax1.bar(x, sunflower_counts, width, label='Sunflowers', color='#FFD700')
    ax1.bar(x, rose_counts, width, bottom=sunflower_counts, label='Roses', color='#FF1493')
    
    ax1.set_xlabel('Dataset Split')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Dataset Distribution by Split')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in splits])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pie chart for class distribution
    total_sunflowers = sum(sunflower_counts)
    total_roses = sum(rose_counts)
    
    ax2.pie([total_sunflowers, total_roses], 
            labels=['Sunflowers', 'Roses'],
            colors=['#FFD700', '#FF1493'],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Overall Class Distribution')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Dataset distribution saved to dataset_distribution.png")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET DISTRIBUTION")
    print("="*60)
    for split in splits:
        total = sum(data[split].values())
        print(f"\n{split.upper()}:")
        print(f"  Sunflowers: {data[split]['sunflowers']} ({data[split]['sunflowers']/total*100:.1f}%)")
        print(f"  Roses: {data[split]['roses']} ({data[split]['roses']/total*100:.1f}%)")
        print(f"  Total: {total}")
    print("="*60)

def save_training_config(args, save_path='training_config.json'):
    """
    Save training configuration to JSON file
    
    Args:
        args: Arguments from argparse
        save_path: Path to save configuration
    """
    config = vars(args)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✓ Training configuration saved to {save_path}")

def load_training_config(config_path='training_config.json'):
    """
    Load training configuration from JSON file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def get_misclassified_images(model, data_loader, device, class_names, save_dir='misclassified'):
    """
    Find and save misclassified images
    
    Args:
        model: Trained model
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        class_names: List of class names
        save_dir: Directory to save misclassified images
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Find misclassified
            mask = predicted != labels
            
            if mask.sum() > 0:
                for i in range(len(labels)):
                    if mask[i]:
                        misclassified.append({
                            'image': images[i].cpu(),
                            'true_label': labels[i].item(),
                            'pred_label': predicted[i].item(),
                            'confidence': torch.softmax(outputs[i], dim=0).max().item()
                        })
    
    print(f"\nFound {len(misclassified)} misclassified images")
    
    # Visualize some misclassified images
    num_to_show = min(len(misclassified), 9)
    
    if num_to_show > 0:
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        for i, ax in enumerate(axes.flat):
            if i < num_to_show:
                item = misclassified[i]
                
                # Denormalize image
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = item['image'] * std + mean
                img_denorm = torch.clamp(img_denorm, 0, 1)
                
                img_np = img_denorm.permute(1, 2, 0).numpy()
                
                ax.imshow(img_np)
                ax.set_title(f"True: {class_names[item['true_label']]}\n"
                           f"Pred: {class_names[item['pred_label']]} "
                           f"({item['confidence']:.2%})",
                           color='red', fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path / 'misclassified_samples.png', dpi=300, bbox_inches='tight')
        print(f"✓ Misclassified samples saved to {save_path / 'misclassified_samples.png'}")
        plt.close()
    
    return misclassified

def compare_models(model_paths, test_loader, device, class_names):
    """
    Compare performance of multiple trained models
    
    Args:
        model_paths: List of model checkpoint paths
        test_loader: DataLoader for test set
        device: Device to run inference on
        class_names: List of class names
    """
    from train import FlowerClassifier
    
    results = {}
    
    for model_path in model_paths:
        print(f"\nEvaluating {model_path}...")
        
        model = FlowerClassifier(pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        results[model_path] = accuracy
        print(f"  Accuracy: {accuracy:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    models = [Path(p).stem for p in model_paths]
    accuracies = list(results.values())
    
    plt.bar(models, accuracies, color=['#FFD700', '#FF1493', '#87CEEB'][:len(models)])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Comparison on Test Set')
    plt.ylim([0, 100])
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Model comparison saved to model_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("Utilities Module for Flower Classification")
    print("="*60)
    print("\nAvailable functions:")
    print("  - visualize_augmentations()")
    print("  - plot_sample_images()")
    print("  - calculate_mean_std()")
    print("  - analyze_dataset_distribution()")
    print("  - get_misclassified_images()")
    print("  - compare_models()")

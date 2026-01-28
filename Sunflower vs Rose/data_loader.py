"""
Data loading and preprocessing utilities
"""

import os
import random
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
import requests
from zipfile import ZipFile
from tqdm import tqdm

class DataLoader:
    """Handle dataset downloading, organization, and loading"""
    
    def __init__(self, data_dir='./data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
    def setup_directories(self):
        """Create directory structure"""
        directories = [
            self.raw_dir / 'sunflowers',
            self.raw_dir / 'roses',
            self.processed_dir / 'train' / 'sunflowers',
            self.processed_dir / 'train' / 'roses',
            self.processed_dir / 'val' / 'sunflowers',
            self.processed_dir / 'val' / 'roses',
            self.processed_dir / 'test' / 'sunflowers',
            self.processed_dir / 'test' / 'roses',
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✓ Directory structure created")
    
    def download_sample_data(self):
        """
        Download sample images for demonstration
        Note: In production, you would use actual flower datasets
        """
        print("\nNote: This is a template for downloading data.")
        print("For real implementation, you can use datasets from:")
        print("1. Kaggle: https://www.kaggle.com/datasets")
        print("2. TensorFlow Datasets")
        print("3. Your own collected images")
        print("\nFor this demo, place your images in:")
        print(f"  - {self.raw_dir / 'sunflowers'}")
        print(f"  - {self.raw_dir / 'roses'}")
    
    def prepare_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Split raw images into train/val/test sets
        
        Args:
            train_ratio: Proportion of training data
            val_ratio: Proportion of validation data
            test_ratio: Proportion of test data
            seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        random.seed(seed)
        
        for flower_type in ['sunflowers', 'roses']:
            raw_flower_dir = self.raw_dir / flower_type
            
            if not raw_flower_dir.exists() or not list(raw_flower_dir.glob('*.jpg')):
                print(f"⚠ Warning: No images found in {raw_flower_dir}")
                continue
            
            # Get all image paths
            image_files = list(raw_flower_dir.glob('*.jpg')) + list(raw_flower_dir.glob('*.png'))
            
            if len(image_files) == 0:
                print(f"⚠ Warning: No images found for {flower_type}")
                continue
            
            print(f"\nProcessing {flower_type}: {len(image_files)} images")
            
            # Shuffle images
            random.shuffle(image_files)
            
            # Calculate split indices
            n_total = len(image_files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            # Split data
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]
            
            # Copy files to respective directories
            self._copy_files(train_files, self.processed_dir / 'train' / flower_type)
            self._copy_files(val_files, self.processed_dir / 'val' / flower_type)
            self._copy_files(test_files, self.processed_dir / 'test' / flower_type)
            
            print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
        
        print("\n✓ Dataset preparation completed!")
        self.print_dataset_statistics()
    
    def _copy_files(self, file_list, dest_dir):
        """Copy files to destination directory"""
        for file_path in file_list:
            dest_path = dest_dir / file_path.name
            shutil.copy2(file_path, dest_path)
    
    def print_dataset_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            split_dir = self.processed_dir / split
            sunflower_count = len(list((split_dir / 'sunflowers').glob('*')))
            rose_count = len(list((split_dir / 'roses').glob('*')))
            total = sunflower_count + rose_count
            
            print(f"\n{split.upper()} SET:")
            print(f"  Sunflowers: {sunflower_count}")
            print(f"  Roses: {rose_count}")
            print(f"  Total: {total}")
        
        print("="*60 + "\n")
    
    def get_data_paths(self, split='train'):
        """
        Get image paths and labels for a specific split
        
        Returns:
            image_paths: List of image file paths
            labels: List of corresponding labels (0: sunflower, 1: rose)
        """
        split_dir = self.processed_dir / split
        
        image_paths = []
        labels = []
        
        # Class 0: Sunflowers
        sunflower_dir = split_dir / 'sunflowers'
        for img_path in sunflower_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_path))
                labels.append(0)
        
        # Class 1: Roses
        rose_dir = split_dir / 'roses'
        for img_path in rose_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_path))
                labels.append(1)
        
        return image_paths, labels

def create_sample_dataset(data_dir='./data', num_samples=50):
    """
    Create a sample dataset with synthetic images for testing
    This is useful when you don't have real images yet
    
    Args:
        data_dir: Directory to create sample data
        num_samples: Number of sample images per class
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    data_path = Path(data_dir)
    raw_sunflowers = data_path / 'raw' / 'sunflowers'
    raw_roses = data_path / 'raw' / 'roses'
    
    raw_sunflowers.mkdir(parents=True, exist_ok=True)
    raw_roses.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} sample images for each class...")
    
    # Create sample sunflower images (yellow-ish)
    for i in tqdm(range(num_samples), desc="Creating sunflowers"):
        img = Image.new('RGB', (224, 224), color=(255, 215, 0))
        draw = ImageDraw.Draw(img)
        
        # Add some random circles to simulate petals
        for _ in range(10):
            x = random.randint(50, 174)
            y = random.randint(50, 174)
            r = random.randint(20, 40)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 200, 0))
        
        # Add center
        draw.ellipse([87, 87, 137, 137], fill=(139, 69, 19))
        
        img.save(raw_sunflowers / f'sunflower_{i:03d}.jpg')
    
    # Create sample rose images (red-ish)
    for i in tqdm(range(num_samples), desc="Creating roses"):
        img = Image.new('RGB', (224, 224), color=(220, 20, 60))
        draw = ImageDraw.Draw(img)
        
        # Add some random circles to simulate petals
        for _ in range(15):
            x = random.randint(50, 174)
            y = random.randint(50, 174)
            r = random.randint(15, 35)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(178, 34, 34))
        
        img.save(raw_roses / f'rose_{i:03d}.jpg')
    
    print(f"\n✓ Created {num_samples} sample images for each class")
    print(f"  Sunflowers: {raw_sunflowers}")
    print(f"  Roses: {raw_roses}")

if __name__ == "__main__":
    # Example usage
    print("Data Loader Module")
    print("="*60)
    
    # Initialize data loader
    loader = DataLoader(data_dir='./data')
    
    # Setup directories
    loader.setup_directories()
    
    # Option 1: Create sample dataset for testing
    create_sample_dataset(num_samples=100)
    
    # Prepare dataset (split into train/val/test)
    loader.prepare_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Get paths for training
    train_paths, train_labels = loader.get_data_paths('train')
    print(f"\nTrain set ready: {len(train_paths)} images")

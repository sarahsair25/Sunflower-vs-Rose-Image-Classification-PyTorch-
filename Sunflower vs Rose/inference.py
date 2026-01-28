"""
Inference script for flower classification
Load trained model and make predictions on new images
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

class FlowerPredictor:
    """Handle model loading and inference"""
    
    def __init__(self, model_path, device=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Sunflower', 'Rose']
        
        # Load model
        self.model = self._load_model()
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load the trained model"""
        from train import FlowerClassifier
        
        print(f"Loading model from {self.model_path}")
        
        # Initialize model
        model = FlowerClassifier(pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded successfully on {self.device}")
        print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        return model
    
    def predict(self, image_path, return_probs=True):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            return_probs: If True, return probabilities, else return class label
            
        Returns:
            If return_probs=True: (predicted_class, confidence, probabilities)
            If return_probs=False: predicted_class
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence = confidence.item()
        
        if return_probs:
            probs = probabilities.cpu().numpy()[0]
            return predicted_class, confidence, probs
        else:
            return predicted_class
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of (image_path, predicted_class, confidence) tuples
        """
        results = []
        
        for img_path in image_paths:
            pred_class, confidence, _ = self.predict(img_path)
            results.append((img_path, pred_class, confidence))
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image and probabilities
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save visualization
        """
        # Get prediction
        pred_class, confidence, probs = self.predict(image_path)
        
        # Load original image
        image = Image.open(image_path)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f'Predicted: {pred_class}\nConfidence: {confidence:.2%}', 
                     fontsize=14, fontweight='bold')
        
        # Display probabilities
        colors = ['#FFD700', '#FF1493']  # Gold for sunflower, pink for rose
        bars = ax2.barh(self.class_names, probs, color=colors, alpha=0.7)
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 1])
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.02, i, f'{prob:.2%}', 
                    va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def batch_visualize(self, image_paths, save_dir='./predictions'):
        """
        Visualize predictions for multiple images
        
        Args:
            image_paths: List of image paths
            save_dir: Directory to save visualizations
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, img_path in enumerate(image_paths):
            output_path = save_path / f'prediction_{i:03d}.png'
            self.visualize_prediction(img_path, save_path=str(output_path))
        
        print(f"\n✓ Visualizations saved to {save_dir}")

def predict_single_image(model_path, image_path):
    """
    Convenient function to predict a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to image
    """
    predictor = FlowerPredictor(model_path)
    pred_class, confidence, probs = predictor.predict(image_path)
    
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    print("\nClass Probabilities:")
    for class_name, prob in zip(predictor.class_names, probs):
        print(f"  {class_name}: {prob:.2%}")
    print("="*60 + "\n")
    
    return pred_class, confidence

def predict_directory(model_path, directory_path, save_results=True):
    """
    Predict all images in a directory
    
    Args:
        model_path: Path to trained model
        directory_path: Directory containing images
        save_results: Whether to save results to CSV
    """
    predictor = FlowerPredictor(model_path)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(directory_path).glob(f'*{ext}'))
    
    if not image_paths:
        print(f"No images found in {directory_path}")
        return
    
    print(f"\nFound {len(image_paths)} images")
    print("Making predictions...\n")
    
    # Make predictions
    results = predictor.predict_batch(image_paths)
    
    # Display results
    print("="*80)
    print(f"{'Image':<40} {'Prediction':<15} {'Confidence'}")
    print("="*80)
    
    for img_path, pred_class, confidence in results:
        print(f"{Path(img_path).name:<40} {pred_class:<15} {confidence:.2%}")
    
    print("="*80)
    
    # Save results if requested
    if save_results:
        import csv
        results_file = Path(directory_path) / 'predictions.csv'
        
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Predicted Class', 'Confidence'])
            
            for img_path, pred_class, confidence in results:
                writer.writerow([Path(img_path).name, pred_class, f'{confidence:.4f}'])
        
        print(f"\n✓ Results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    print("Flower Classification - Inference Module")
    print("="*60)
    
    # Example usage
    model_path = 'best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"\n⚠ Model file not found: {model_path}")
        print("Please train the model first using main.py")
    else:
        print(f"\nModel found: {model_path}")
        print("\nUsage examples:")
        print("1. Predict single image:")
        print("   predict_single_image('best_model.pth', 'path/to/image.jpg')")
        print("\n2. Predict directory of images:")
        print("   predict_directory('best_model.pth', 'path/to/images/')")

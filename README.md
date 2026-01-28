
<img width="1536" height="1024" alt="Rose vs sunflower_ image classification" src="https://github.com/user-attachments/assets/0b0a6388-2d6f-4d05-baec-22bcd9a87a41" />

<div align="center">

# 🌻🌹 Sunflower vs Rose Image Classifier


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)

A complete PyTorch implementation of binary image classification using deep learning to distinguish between sunflower and rose images.

[Features](#-features) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Results](#-results)

<img src="https://img.shields.io/badge/Transfer_Learning-ResNet18-orange" />
<img src="https://img.shields.io/badge/Accuracy-90%25+-success" />

</div>

---

## 🌟 Overview

This project implements a **binary image classification system** to distinguish between sunflower and rose images using PyTorch. It showcases modern deep learning practices including transfer learning, data augmentation, and comprehensive model evaluation.

### Why This Project?

- ✅ **Educational**: Perfect for learning PyTorch and computer vision
- ✅ **Production-Ready**: Complete pipeline from data to deployment
- ✅ **Well-Documented**: Extensive documentation and examples
- ✅ **Customizable**: Easy to adapt for other classification tasks

---

## ✨ Features

### Core Functionality
- 🎯 **Binary Image Classification** - Sunflower vs Rose detection
- 🚀 **Transfer Learning** - Pre-trained ResNet18 for fast training
- 🎨 **Data Augmentation** - Comprehensive augmentation pipeline
- 📊 **Visualization Tools** - Training curves, confusion matrices, predictions
- 💾 **Model Checkpointing** - Automatic best model saving
- ⚡ **GPU Acceleration** - CUDA support for faster training

### Model Options
- **ResNet18** (Transfer Learning) - Recommended for best accuracy
- **Custom CNN** (From Scratch) - Lightweight alternative

### Advanced Features
- Batch prediction capabilities
- Misclassification analysis
- Model comparison tools
- Interactive Jupyter notebook
- Comprehensive test suite

---

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, recommended for faster training)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/sunflower-rose-classifier.git
cd sunflower-rose-classifier
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python test_setup.py
```

---

## 🚀 Quick Start

### Option 1: Train with Sample Data (No images needed!)

Perfect for testing and learning:

```bash
python main.py --create_sample --num_samples 100 --num_epochs 10
```

This will:
- ✅ Generate 100 synthetic images per class
- ✅ Split data into train/val/test sets (70/15/15)
- ✅ Train a ResNet18 model
- ✅ Generate evaluation metrics
- ✅ Save the best model

### Option 2: Train with Your Own Data

1. **Organize your images:**
```
data/raw/sunflowers/    # Place sunflower images here
data/raw/roses/         # Place rose images here
```

2. **Prepare the dataset:**
```bash
python data_loader.py
```

3. **Train the model:**
```bash
python main.py --mode train --num_epochs 20 --batch_size 32
```

### Make Predictions

```bash
# Predict single image
python main.py --mode predict --predict_path path/to/image.jpg

# Predict directory
python main.py --mode predict --predict_path path/to/images/
```
-## 💻 Usage

### Training

#### Basic Training
```bash
python main.py --mode train --num_epochs 20
```

#### Advanced Options
```bash
python main.py \
    --mode train \
    --model_type resnet \
    --pretrained \
    --num_epochs 30 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --image_size 224
```

#### Train Custom CNN (No Transfer Learning)
```bash
python main.py --mode train --model_type simple --num_epochs 50
```

### Evaluation

Evaluate on test set:
```bash
python main.py --mode eval --model_path best_model.pth
```

### Inference

#### Python API
```python
from inference import FlowerPredictor

# Initialize predictor
predictor = FlowerPredictor('best_model.pth')

# Predict single image
pred_class, confidence, probs = predictor.predict('test_image.jpg')
print(f"Prediction: {pred_class} ({confidence:.2%} confidence)")

# Visualize prediction
predictor.visualize_prediction('test_image.jpg', save_path='result.png')

# Batch prediction
results = predictor.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

#### Command Line
```bash
# Single image
python main.py --mode predict --predict_path flower.jpg

# Directory of images
python main.py --mode predict --predict_path ./test_images/
```

---

## 🏗️ Model Architecture

### ResNet18 (Transfer Learning) - Default

```
Input (224x224x3)
    ↓
ResNet18 Backbone (Pre-trained on ImageNet)
    ↓
Custom Classifier Head:
    ├─ Linear(512 → 512)
    ├─ ReLU
    ├─ Dropout(0.5)
    └─ Linear(512 → 2)
    ↓
Output (Sunflower/Rose)
```

**Parameters**: ~11M trainable  
**Pre-training**: ImageNet dataset  
**Accuracy**: 90-95% (with adequate data)

### Simple CNN (From Scratch) - Alternative

```
Input (224x224x3)
    ↓
4x [Conv2d → BatchNorm → ReLU → MaxPool]
    ↓
Adaptive Average Pooling
    ↓
Fully Connected Layers
    ↓
Output (2 classes)
```

**Parameters**: ~2M trainable  
**Training**: From random initialization  
**Accuracy**: 85-90% (with adequate data)

---

## 📊 Results

### Expected Performance

With 100+ images per class:

| Metric | ResNet18 | Simple CNN |
|--------|----------|------------|
| Train Accuracy | 95-99% | 90-95% |
| Val Accuracy | 90-95% | 85-90% |
| Test Accuracy | 88-93% | 83-88% |
| Training Time* | 5-10 min | 10-20 min |

*On NVIDIA GPU with 100 images per class, 20 epochs

### Sample Outputs

The project generates:

1. **Training Metrics**
   - Loss curves (train/val)
   - Accuracy curves (train/val)

2. **Confusion Matrix**
   - True positives/negatives
   - False positives/negatives

3. **Classification Report**
   - Precision, Recall, F1-Score
   - Per-class metrics

4. **Prediction Visualizations**
   - Original image
   - Predicted class
   - Confidence scores

---

### Key Topics Covered

- Transfer Learning with PyTorch
- Data Augmentation Techniques
- Model Training and Validation
- Evaluation Metrics
- Inference Pipeline
- Best Practices

---

## 🎓 Learning Resources

### Concepts Demonstrated

- ✅ Transfer Learning
- ✅ Binary Classification
- ✅ Data Augmentation
- ✅ Model Checkpointing
- ✅ Learning Rate Scheduling
- ✅ Batch Processing
- ✅ GPU Acceleration

### Code Examples

The project includes extensive examples for:
- Custom Dataset implementation
- Model definition and training
- Data preprocessing pipelines
- Prediction and visualization
- Model evaluation

---

## 🛠️ Command Reference

### Common Commands

```bash
# Create sample dataset and train
python main.py --create_sample --num_epochs 10

# Train with custom settings
python main.py --mode train --batch_size 64 --num_epochs 30

# Evaluate model
python main.py --mode eval

# Make predictions
python main.py --mode predict --predict_path image.jpg

# Run tests
python test_setup.py
```

### All Available Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Data directory |
| `--create_sample` | False | Create sample dataset |
| `--num_samples` | 100 | Samples per class |
| `--batch_size` | 32 | Training batch size |
| `--num_epochs` | 20 | Training epochs |
| `--learning_rate` | 0.001 | Learning rate |
| `--image_size` | 224 | Input image size |
| `--model_type` | `resnet` | Model (resnet/simple) |
| `--pretrained` | True | Use pretrained weights |
| `--mode` | `train` | train/eval/predict |
| `--model_path` | `best_model.pth` | Model file path |
| `--predict_path` | None | Image/directory path |

---

## 🔬 Advanced Usage

### Custom Data Augmentation

Modify `get_transforms()` in `train.py`:

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

### Mixed Precision Training

Enable for faster training on compatible GPUs:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Export to ONNX

```python
import torch
from train import FlowerClassifier

model = FlowerClassifier(pretrained=False)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "flower_classifier.onnx")
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create your feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🎨 UI/Visualization enhancements
- 🧪 Additional test cases
- 🌍 Multi-class classification support

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python main.py --batch_size 16
```

**2. Low Accuracy**
- Ensure balanced classes (equal images per class)
- Use at least 100 images per class
- Enable data augmentation
- Increase training epochs
- Use transfer learning (`--model_type resnet --pretrained`)

**3. Slow Training**
- Check GPU availability: `python test_setup.py`
- Increase batch size if memory allows
- Use more data loading workers: `--num_workers 4`

**4. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Sarah Sair

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🙏 Acknowledgments

- **PyTorch Team** - For the excellent deep learning framework
- **ResNet Authors** - Deep Residual Learning for Image Recognition (He et al., 2015)
- **ImageNet** - For pre-trained model weights
- **Community Contributors** - For feedback and improvements

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/sunflower-rose-classifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/sunflower-rose-classifier/discussions)


## 🎯 Use Cases

This project can be adapted for:

- 🌸 Flower species classification
- 🍃 Plant disease detection
- 🏥 Medical image analysis
- 🏭 Quality control in manufacturing
- 🎨 Art style classification
- 📚 Educational purposes

---

## 🗺️ Roadmap

Future enhancements planned:

- [ ] Multi-class classification support
- [ ] Web interface for predictions
- [ ] Mobile app deployment
- [ ] Real-time video classification
- [ ] Model quantization for edge devices
- [ ] Docker containerization
- [ ] REST API implementation
- [ ] Grad-CAM visualization

---

## 📈 Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/sunflower-rose-classifier?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/sunflower-rose-classifier?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/sunflower-rose-classifier?style=social)

---

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

Made with ❤️ and PyTorch

[Report Bug](https://github.com/yourusername/sunflower-rose-classifier/issues) • [Request Feature](https://github.com/yourusername/sunflower-rose-classifier/issues)

</div>

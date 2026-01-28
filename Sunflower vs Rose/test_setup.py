"""
Test script to verify project installation and setup
Run this to check if everything is working correctly
"""

import sys
import importlib

def check_imports():
    """Check if all required packages are installed"""
    print("="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ {name:20s} - Installed")
        except ImportError:
            print(f"✗ {name:20s} - NOT INSTALLED")
            missing_packages.append(name)
    
    print("="*70)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True

def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*70)
    print("CHECKING GPU SUPPORT")
    print("="*70)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("✗ CUDA is not available")
            print("  Training will use CPU (slower)")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
    
    print("="*70)

def check_project_structure():
    """Check if all project files are present"""
    print("\n" + "="*70)
    print("CHECKING PROJECT STRUCTURE")
    print("="*70)
    
    required_files = [
        'main.py',
        'train.py',
        'data_loader.py',
        'inference.py',
        'utils.py',
        'requirements.txt',
        'README.md',
        'DOCUMENTATION.md'
    ]
    
    import os
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")
            missing_files.append(file)
    
    print("="*70)
    
    if missing_files:
        print(f"\n⚠ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n✓ All project files are present!")
        return True

def test_imports():
    """Test importing project modules"""
    print("\n" + "="*70)
    print("TESTING PROJECT MODULES")
    print("="*70)
    
    modules = [
        ('train', ['FlowerDataset', 'FlowerClassifier', 'Trainer']),
        ('data_loader', ['DataLoader', 'create_sample_dataset']),
        ('inference', ['FlowerPredictor']),
        ('utils', ['visualize_augmentations', 'plot_sample_images'])
    ]
    
    all_success = True
    
    for module_name, components in modules:
        try:
            module = importlib.import_module(module_name)
            print(f"\n✓ Module '{module_name}' imported successfully")
            
            for component in components:
                if hasattr(module, component):
                    print(f"  ✓ {component}")
                else:
                    print(f"  ✗ {component} not found")
                    all_success = False
                    
        except Exception as e:
            print(f"\n✗ Error importing module '{module_name}': {e}")
            all_success = False
    
    print("="*70)
    
    if all_success:
        print("\n✓ All modules imported successfully!")
    else:
        print("\n⚠ Some modules had errors")
    
    return all_success

def run_quick_test():
    """Run a quick functionality test"""
    print("\n" + "="*70)
    print("RUNNING QUICK FUNCTIONALITY TEST")
    print("="*70)
    
    try:
        import torch
        from train import FlowerClassifier
        
        print("\n1. Testing model initialization...")
        model = FlowerClassifier(pretrained=False)
        print("   ✓ Model created successfully")
        
        print("\n2. Testing forward pass...")
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"   ✓ Forward pass successful - Output shape: {output.shape}")
        
        if output.shape == torch.Size([1, 2]):
            print("   ✓ Output shape is correct [1, 2]")
        else:
            print(f"   ✗ Unexpected output shape: {output.shape}")
        
        print("\n3. Testing data transformation...")
        from train import get_transforms
        train_transform, val_transform = get_transforms()
        print("   ✓ Transforms created successfully")
        
        print("\n" + "="*70)
        print("✓ QUICK TEST PASSED!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during quick test: {e}")
        print("="*70)
        return False

def main():
    """Main test function"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "SUNFLOWER VS ROSE CLASSIFIER" + " "*25 + "║")
    print("║" + " "*20 + "INSTALLATION TEST" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Run all checks
    deps_ok = check_imports()
    check_cuda()
    structure_ok = check_project_structure()
    
    if deps_ok and structure_ok:
        imports_ok = test_imports()
        
        if imports_ok:
            test_ok = run_quick_test()
            
            if test_ok:
                print("\n" + "🎉 "*35)
                print("\n" + " "*20 + "ALL TESTS PASSED!")
                print("\n" + " "*10 + "You're ready to start training your model!")
                print("\n" + "Quick start commands:")
                print("  - Create sample data and train:")
                print("    python main.py --create_sample --num_epochs 10")
                print("\n  - Train with your own data:")
                print("    python main.py --mode train --num_epochs 20")
                print("\n" + "🎉 "*35 + "\n")
                return True
    
    print("\n" + "⚠ "*35)
    print("\nSome tests failed. Please fix the issues above before proceeding.")
    print("\n" + "⚠ "*35 + "\n")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

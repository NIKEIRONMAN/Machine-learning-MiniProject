"""
Quick test script to verify the installation and data
"""
import os
import sys

def check_installation():
    """Check if all required packages are installed"""
    print("="*60)
    print("CHECKING INSTALLATION")
    print("="*60)
    
    required_packages = [
        'numpy',
        'pandas',
        'sklearn',
        'joblib',
        'streamlit',
        'plotly',
        'matplotlib',
        'PIL',
        'cv2',
        'scipy',
        'seaborn',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️ Missing packages found!")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All packages installed successfully!")
        return True


def check_data():
    """Check if data files exist"""
    print("\n" + "="*60)
    print("CHECKING DATA FILES")
    print("="*60)
    
    data_files = [
        'data/train-images.idx3-ubyte',
        'data/train-labels.idx1-ubyte',
        'data/t10k-images.idx3-ubyte',
        'data/t10k-labels.idx1-ubyte'
    ]
    
    xray_dir = 'data/Spot on Bare Metal_XrayImages_Raw'
    
    all_found = True
    
    for file in data_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {file} - NOT FOUND")
            all_found = False
    
    # Check X-ray directory
    if os.path.exists(xray_dir):
        tif_files = [f for f in os.listdir(xray_dir) if f.endswith('.tif')]
        print(f"✓ {xray_dir} ({len(tif_files)} TIFF files)")
    else:
        print(f"✗ {xray_dir} - NOT FOUND")
        all_found = False
    
    if all_found:
        print("\n✅ All data files found!")
    else:
        print("\n⚠️ Some data files are missing!")
        print("Make sure Mechanical MNIST data is in the data/ directory")
    
    return all_found


def check_structure():
    """Check project directory structure"""
    print("\n" + "="*60)
    print("CHECKING PROJECT STRUCTURE")
    print("="*60)
    
    required_dirs = [
        'data',
        'src',
        'app',
        'models',
        'notebooks'
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - NOT FOUND")
            os.makedirs(dir_name, exist_ok=True)
            print(f"  → Created {dir_name}/")
    
    print("\n✅ Project structure verified!")


def test_data_loading():
    """Test loading a small sample of data"""
    print("\n" + "="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    try:
        from src.load_mnist_data import load_mechanical_mnist
        
        print("Loading Mechanical MNIST data...")
        X_train, y_train, X_test, y_test = load_mechanical_mnist()
        
        print(f"✓ Training data: {X_train.shape}")
        print(f"✓ Test data: {X_test.shape}")
        print(f"✓ Property range: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
        print("\n✅ Data loading successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        return False


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("ML-ASSISTED METASTRUCTURE DESIGNER")
    print("INSTALLATION & DATA VERIFICATION")
    print("="*60 + "\n")
    
    # Check installation
    install_ok = check_installation()
    
    # Check structure
    check_structure()
    
    # Check data
    data_ok = check_data()
    
    # Test data loading if everything is OK
    if install_ok and data_ok:
        test_ok = test_data_loading()
    else:
        test_ok = False
    
    # Final summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if install_ok and data_ok and test_ok:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou can now:")
        print("1. Train the model: python train_model.py")
        print("2. Run the app: streamlit run app/app.py")
        print("3. Visualize data: python src/visualize_data.py")
    else:
        print("⚠️ SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before proceeding.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Ensure Mechanical MNIST data is in data/ directory")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

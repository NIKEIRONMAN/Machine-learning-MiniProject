"""
Load Mechanical MNIST data from IDX format
Based on the standard MNIST data format
"""
import numpy as np
import struct
import os


def read_idx_images(filename):
    """
    Read images from IDX3-UBYTE format
    
    Returns:
        numpy array of shape (n_samples, height, width)
    """
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('>I', f.read(4))[0]
        n_images = struct.unpack('>I', f.read(4))[0]
        n_rows = struct.unpack('>I', f.read(4))[0]
        n_cols = struct.unpack('>I', f.read(4))[0]
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n_images, n_rows, n_cols)
        
    return images


def read_idx_labels(filename):
    """
    Read labels from IDX1-UBYTE format
    
    Returns:
        numpy array of shape (n_samples,)
    """
    with open(filename, 'rb') as f:
        # Read header
        magic = struct.unpack('>I', f.read(4))[0]
        n_labels = struct.unpack('>I', f.read(4))[0]
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels


def load_mechanical_mnist(data_dir='data'):
    """
    Load Mechanical MNIST dataset
    
    Returns:
        X_train: Training images
        y_train: Training labels (mechanical properties)
        X_test: Test images
        y_test: Test labels
    """
    # File paths
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    
    # Load data
    print("Loading Mechanical MNIST dataset...")
    X_train = read_idx_images(train_images_path)
    y_train = read_idx_labels(train_labels_path)
    X_test = read_idx_images(test_images_path)
    y_test = read_idx_labels(test_labels_path)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Image size: {X_train.shape[1]}x{X_train.shape[2]}")
    
    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # For mechanical MNIST, labels represent mechanical property classes
    # Convert to continuous values (simplified - adjust based on actual data)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # Test loading
    X_train, y_train, X_test, y_test = load_mechanical_mnist()
    print(f"\nData loaded successfully!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Label range: {y_train.min():.2f} - {y_train.max():.2f}")

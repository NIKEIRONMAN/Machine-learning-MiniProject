"""
Visualization utilities for Mechanical MNIST data
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.load_mnist_data import load_mechanical_mnist
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def visualize_sample_images(X, y, n_samples=16, save_path=None):
    """
    Visualize sample images from the dataset
    
    Args:
        X: Image data (n_samples, height, width)
        y: Labels/properties
        n_samples: Number of samples to display
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Sample Topology Images from Mechanical MNIST', fontsize=16, fontweight='bold')
    
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        i = indices[idx]
        ax.imshow(X[i], cmap='gray')
        ax.set_title(f'Property: {y[i]:.2f}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_property_distribution(y_train, y_test, save_path=None):
    """
    Plot distribution of mechanical properties
    
    Args:
        y_train: Training labels
        y_test: Test labels
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(y_train, bins=30, alpha=0.6, label='Training', color='blue', edgecolor='black')
    axes[0].hist(y_test, bins=30, alpha=0.6, label='Test', color='orange', edgecolor='black')
    axes[0].set_xlabel('Mechanical Property Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Mechanical Properties', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [y_train, y_test]
    axes[1].boxplot(data_to_plot, labels=['Training', 'Test'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Mechanical Property Value', fontsize=12)
    axes[1].set_title('Property Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_image_statistics(X, save_path=None):
    """
    Plot statistics of image data
    
    Args:
        X: Image data (n_samples, height, width)
        save_path: Path to save figure (optional)
    """
    # Calculate statistics
    mean_intensities = X.mean(axis=(1, 2))
    std_intensities = X.std(axis=(1, 2))
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Average image
    avg_image = X.mean(axis=0)
    im = axes[0].imshow(avg_image, cmap='viridis')
    axes[0].set_title('Average Topology Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Mean intensity distribution
    axes[1].hist(mean_intensities, bins=50, color='green', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Mean Pixel Intensity', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Mean Intensities', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Std intensity distribution
    axes[2].hist(std_intensities, bins=50, color='purple', edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Std Dev of Pixel Intensity', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Distribution of Intensity Std Dev', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def create_eda_report(save_dir='notebooks/figures'):
    """
    Create complete EDA report with visualizations
    
    Args:
        save_dir: Directory to save figures
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading Mechanical MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mechanical_mnist()
    
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("="*60)
    
    # Dataset statistics
    print("\n1. Dataset Statistics:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Image dimensions: {X_train.shape[1]} x {X_train.shape[2]}")
    print(f"   Property range (train): [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"   Property range (test): [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"   Property mean (train): {y_train.mean():.2f} ± {y_train.std():.2f}")
    
    # Generate visualizations
    print("\n2. Generating visualizations...")
    
    print("   - Sample images...")
    visualize_sample_images(
        X_train, y_train,
        save_path=os.path.join(save_dir, 'sample_images.png')
    )
    
    print("   - Property distributions...")
    plot_property_distribution(
        y_train, y_test,
        save_path=os.path.join(save_dir, 'property_distribution.png')
    )
    
    print("   - Image statistics...")
    plot_image_statistics(
        X_train,
        save_path=os.path.join(save_dir, 'image_statistics.png')
    )
    
    print("\n" + "="*60)
    print("✅ EDA REPORT COMPLETE!")
    print(f"Figures saved to: {save_dir}/")
    print("="*60)


if __name__ == '__main__':
    create_eda_report()

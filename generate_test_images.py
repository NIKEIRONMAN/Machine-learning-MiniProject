"""
Generate dummy/example topology images for testing the app
"""
import numpy as np
from PIL import Image
import os

# Create examples directory
EXAMPLES_DIR = 'examples'
os.makedirs(EXAMPLES_DIR, exist_ok=True)


def create_grid_pattern(size=28, density=0.3):
    """Create a grid/lattice pattern"""
    img = np.zeros((size, size))
    spacing = int(size * (1 - density) / 4)
    
    for i in range(0, size, spacing):
        img[i, :] = 255
        img[:, i] = 255
    
    return img.astype(np.uint8)


def create_honeycomb_pattern(size=28, density=0.4):
    """Create a honeycomb-like pattern"""
    img = np.zeros((size, size))
    
    for i in range(0, size, 4):
        for j in range(0, size, 4):
            if (i + j) % 8 == 0:
                img[i:i+2, j:j+2] = 255
    
    return img.astype(np.uint8)


def create_diagonal_pattern(size=28, density=0.5):
    """Create diagonal strut pattern"""
    img = np.zeros((size, size))
    
    for i in range(size):
        img[i, i] = 255
        img[i, size-1-i] = 255
        
    # Add thickness
    for i in range(1, 3):
        for j in range(size):
            if j + i < size:
                img[j, j+i] = 255
                img[j+i, j] = 255
            if j - i >= 0:
                img[j, size-1-j+i] = 255 if size-1-j+i < size else 0
    
    return img.astype(np.uint8)


def create_random_porous(size=28, density=0.6):
    """Create random porous structure"""
    img = (np.random.rand(size, size) < density) * 255
    return img.astype(np.uint8)


def create_gyroid_like(size=28):
    """Create a gyroid-inspired pattern"""
    img = np.zeros((size, size))
    
    x = np.linspace(0, 2*np.pi, size)
    y = np.linspace(0, 2*np.pi, size)
    X, Y = np.meshgrid(x, y)
    
    # Simplified gyroid approximation
    Z = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(X)
    img = ((Z > -0.3) & (Z < 0.3)) * 255
    
    return img.astype(np.uint8)


def create_octet_like(size=28):
    """Create octet truss-like pattern"""
    img = np.zeros((size, size))
    
    # Vertical and horizontal struts
    for i in range(0, size, 7):
        img[i, :] = 255
        img[:, i] = 255
    
    # Diagonal struts
    for i in range(0, size, 7):
        for j in range(max(0, i-2), min(size, i+3)):
            if j < size:
                img[j, j] = 255
    
    return img.astype(np.uint8)


def generate_all_examples():
    """Generate all example images"""
    
    print("="*60)
    print("GENERATING EXAMPLE TOPOLOGY IMAGES")
    print("="*60)
    
    examples = {
        'grid_low_density.png': create_grid_pattern(28, 0.2),
        'grid_medium_density.png': create_grid_pattern(28, 0.4),
        'grid_high_density.png': create_grid_pattern(28, 0.6),
        'honeycomb.png': create_honeycomb_pattern(28, 0.4),
        'diagonal_struts.png': create_diagonal_pattern(28, 0.5),
        'random_porous_low.png': create_random_porous(28, 0.3),
        'random_porous_high.png': create_random_porous(28, 0.7),
        'gyroid_like.png': create_gyroid_like(28),
        'octet_like.png': create_octet_like(28),
    }
    
    for filename, img_array in examples.items():
        img = Image.fromarray(img_array, mode='L')
        filepath = os.path.join(EXAMPLES_DIR, filename)
        img.save(filepath)
        print(f"✓ Created: {filename}")
    
    # Create a larger version for better visualization
    print("\nCreating high-resolution versions...")
    for filename, img_array in examples.items():
        img = Image.fromarray(img_array, mode='L')
        img_large = img.resize((256, 256), Image.NEAREST)
        filepath = os.path.join(EXAMPLES_DIR, filename.replace('.png', '_large.png'))
        img_large.save(filepath)
    
    print(f"\n✅ Generated {len(examples)} example images")
    print(f"Saved to: {EXAMPLES_DIR}/")
    print("\nThese images can be used to test the Streamlit app!")
    print("="*60)


if __name__ == '__main__':
    generate_all_examples()

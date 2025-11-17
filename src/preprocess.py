"""
Preprocessing pipeline for X-ray images of additive manufacturing metastructures
"""
import numpy as np
import cv2
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def load_xray_image(image_path, target_size=(256, 256)):
    """
    Load and preprocess X-ray TIFF image
    
    Args:
        image_path: Path to TIFF image file
        target_size: Target size for resizing (height, width)
    
    Returns:
        Normalized image array
    """
    try:
        # Load TIFF image
        img = Image.open(image_path)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Resize to target size
        img_resized = cv2.resize(img_array, target_size[::-1])
        
        # Normalize to [0, 1]
        if img_resized.max() > 0:
            img_normalized = img_resized / img_resized.max()
        else:
            img_normalized = img_resized
        
        return img_normalized
    
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def extract_features_from_image(img_array):
    """
    Extract engineered features from X-ray image
    
    Features include:
    - Statistical features (mean, std, min, max)
    - Texture features (entropy, contrast)
    - Geometric features (edge density, porosity estimate)
    
    Args:
        img_array: Normalized image array
    
    Returns:
        Feature vector (1D numpy array)
    """
    features = []
    
    # Convert to uint8 for some operations
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    # Statistical features
    features.append(np.mean(img_array))
    features.append(np.std(img_array))
    features.append(np.min(img_array))
    features.append(np.max(img_array))
    features.append(np.median(img_array))
    
    # Texture features using histogram
    hist, _ = np.histogram(img_array, bins=32)
    hist = hist / hist.sum()  # Normalize
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    features.append(entropy)
    
    # Edge detection for porosity/structure analysis
    edges = cv2.Canny(img_uint8, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)
    
    # Local Binary Pattern-like feature
    contrast = np.std(img_array) / (np.mean(img_array) + 1e-10)
    features.append(contrast)
    
    # Intensity distribution (quartiles)
    q25, q50, q75 = np.percentile(img_array, [25, 50, 75])
    features.append(q25)
    features.append(q50)
    features.append(q75)
    
    # Roughness indicator
    gradient_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    features.append(np.mean(gradient_magnitude))
    features.append(np.std(gradient_magnitude))
    
    return np.array(features)


def parse_filename_parameters(filename):
    """
    Extract process parameters from filename
    
    Example: 015_Ti64_P23t2077um_S2F2.8mm_U18G1200001.tif
    P = Power, S = Scan speed, F = Feature size, etc.
    
    Args:
        filename: Image filename
    
    Returns:
        Dictionary of extracted parameters
    """
    params = {
        'material': 'Ti64',
        'power_w': 230,  # Default values - update based on filename pattern
        'spot_size_um': 2077,
        'scan_speed_mm_s': 2.0,
        'feature_size_mm': 2.8,
        'energy_density': 0.0
    }
    
    try:
        # Parse filename for parameters
        parts = filename.split('_')
        
        for part in parts:
            # Power extraction: P23t2077um -> Power ~23
            if 'P' in part and 't' in part:
                power_str = part.split('P')[1].split('t')[0]
                params['power_w'] = float(power_str) * 10  # Scale appropriately
                
                spot_str = part.split('t')[1].replace('um', '')
                params['spot_size_um'] = float(spot_str)
            
            # Scan speed and feature size: S2F2.8mm
            if 'S' in part and 'F' in part:
                scan_str = part.split('S')[1].split('F')[0]
                params['scan_speed_mm_s'] = float(scan_str)
                
                feature_str = part.split('F')[1].replace('mm', '')
                params['feature_size_mm'] = float(feature_str)
        
        # Calculate energy density (simplified)
        if params['scan_speed_mm_s'] > 0 and params['spot_size_um'] > 0:
            params['energy_density'] = params['power_w'] / (params['scan_speed_mm_s'] * params['spot_size_um'] / 1000)
    
    except Exception as e:
        print(f"Warning: Could not parse filename {filename}: {e}")
    
    return params


def load_dataset_from_images(image_dir, target_size=(256, 256), use_features=True):
    """
    Load all X-ray images from directory and create dataset
    
    Args:
        image_dir: Directory containing TIFF images
        target_size: Target image size
        use_features: If True, extract engineered features; if False, use flattened pixels
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        metadata: List of dictionaries with filename and process parameters
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    
    X_list = []
    metadata_list = []
    
    print(f"Loading {len(image_files)} images from {image_dir}...")
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        
        # Load and preprocess image
        img_array = load_xray_image(img_path, target_size)
        
        if img_array is not None:
            if use_features:
                # Extract engineered features
                features = extract_features_from_image(img_array)
                X_list.append(features)
            else:
                # Use flattened pixel values
                X_list.append(img_array.flatten())
            
            # Parse filename for process parameters
            params = parse_filename_parameters(img_file)
            params['filename'] = img_file
            metadata_list.append(params)
    
    X = np.array(X_list)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, metadata_list


def prepare_input(df_input, ohe, scaler, num_cols, cat_cols):
    """
    Prepare input for prediction (legacy function for compatibility)
    """
    X_num = df_input[num_cols].astype(float).values
    X_num = scaler.transform(X_num)
    X_cat = ohe.transform(df_input[cat_cols])
    X_proc = np.hstack([X_num, X_cat])
    return X_proc
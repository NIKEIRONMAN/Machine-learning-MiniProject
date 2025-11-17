"""
Quick prediction script for testing trained model
"""
import joblib
import numpy as np
import json
import os
from PIL import Image


def load_model():
    """Load trained model and artifacts"""
    model = joblib.load('models/rf_modulus_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    with open('models/meta.json', 'r') as f:
        meta = json.load(f)
    
    return model, scaler, meta


def predict_from_image(image_path, model, scaler, target_size=(28, 28)):
    """
    Make prediction from image file
    
    Args:
        image_path: Path to image file
        model: Trained model
        scaler: Feature scaler
        target_size: Target image dimensions
    
    Returns:
        Predicted value
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Flatten and scale
    features = img_array.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    return prediction


def predict_from_array(img_array, model, scaler):
    """
    Make prediction from numpy array
    
    Args:
        img_array: Image as numpy array (height, width)
        model: Trained model
        scaler: Feature scaler
    
    Returns:
        Predicted value
    """
    # Flatten and scale
    features = img_array.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    return prediction


def batch_predict(image_paths, model, scaler, target_size=(28, 28)):
    """
    Make predictions for multiple images
    
    Args:
        image_paths: List of image file paths
        model: Trained model
        scaler: Feature scaler
        target_size: Target image dimensions
    
    Returns:
        List of predictions
    """
    predictions = []
    
    for img_path in image_paths:
        try:
            pred = predict_from_image(img_path, model, scaler, target_size)
            predictions.append({
                'file': os.path.basename(img_path),
                'prediction': float(pred)
            })
        except Exception as e:
            predictions.append({
                'file': os.path.basename(img_path),
                'error': str(e)
            })
    
    return predictions


def main():
    """Example usage"""
    print("="*60)
    print("QUICK PREDICTION TOOL")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model, scaler, meta = load_model()
    
    print(f"✓ Model loaded: {meta.get('model_type', 'Unknown')}")
    print(f"✓ Target property: {meta.get('target', 'Unknown')}")
    print(f"✓ R² Score: {meta.get('r2_score', 0):.4f}")
    
    # Test with sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (from test set)")
    print("="*60)
    
    with open('models/sample_predictions.json', 'r') as f:
        samples = json.load(f)
    
    print("\n{:<15} {:<15} {:<15}".format("True Value", "Predicted", "Error"))
    print("-" * 45)
    
    for true_val, pred_val in zip(samples['y_true'], samples['y_pred']):
        error = abs(true_val - pred_val)
        print("{:<15.2f} {:<15.2f} {:<15.2f}".format(true_val, pred_val, error))
    
    print("\n" + "="*60)
    print("To use this script for your own images:")
    print("="*60)
    print("""
from predict import load_model, predict_from_image

# Load model
model, scaler, meta = load_model()

# Predict from image
prediction = predict_from_image('path/to/image.tif', model, scaler)
print(f"Predicted {meta['target']}: {prediction:.2f}")
    """)


if __name__ == '__main__':
    main()

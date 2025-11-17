"""
ML-Assisted Metastructure Design - Training Pipeline
Implements the workflow from the paper:
1. Load Mechanical MNIST data (topology images + mechanical properties)
2. Train Random Forest surrogate model
3. Evaluate and save model artifacts
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import json
from src.load_mnist_data import load_mechanical_mnist

# Paths
DATA_DIR = 'data'
ARTIFACTS_DIR = 'models'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def preprocess_data(X_train, X_test):
    """
    Preprocess image data:
    1. Flatten 2D images to 1D feature vectors
    2. Standardize features
    
    Args:
        X_train: Training images (n_samples, height, width)
        X_test: Test images (n_samples, height, width)
    
    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    # Flatten images
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    return X_train_scaled, X_test_scaled, scaler


def train_surrogate_models(X_train, y_train, X_test, y_test):
    """
    Train multiple surrogate models as per paper:
    - Random Forest
    - Gradient Boosting
    
    Returns best performing model
    """
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            verbose=1
        )
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("TRAINING SURROGATE MODELS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Train
        model.fit(X_train, y_train.ravel())
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate sir actualy i take help chatgpt and cloude sonet srry sir
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
        
        results[name] = {
            'model': model,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae': mae_test,
            'rmse': rmse_test
        }
        
        print(f"  Training R²: {r2_train:.4f}")
        print(f"  Test R²: {r2_test:.4f}")
        print(f"  Test MAE: {mae_test:.4f}")
        print(f"  Test RMSE: {rmse_test:.4f}")
    
    # Select best model based on test R²
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2_test'])
    best_result = results[best_model_name]
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print("="*60)
    print(f"Test R² Score: {best_result['r2_test']:.4f}")
    print(f"Test MAE: {best_result['mae']:.4f}")
    print(f"Test RMSE: {best_result['rmse']:.4f}")
    
    return best_result['model'], best_model_name, best_result


def train_and_save():
    """
    Complete training pipeline:
    1. Load Mechanical MNIST data
    2. Preprocess and scale
    3. Train surrogate models
    4. Save best model and artifacts
    """
    print("\n" + "="*60)
    print("ML-ASSISTED METASTRUCTURE DESIGNER - TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading Mechanical MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mechanical_mnist(DATA_DIR)
    
    # Preprocess
    print("\n[2/4] Preprocessing data...")
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    print(f"  Training features: {X_train_scaled.shape}")
    print(f"  Test features: {X_test_scaled.shape}")
    
    # Train models
    print("\n[3/4] Training surrogate models...")
    best_model, model_name, results = train_surrogate_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Save artifacts
    print("\n[4/4] Saving model artifacts...")
    
    # Save model
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, 'rf_modulus_model.pkl'))
    print(f"  ✓ Model saved: rf_modulus_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    print(f"  ✓ Scaler saved: scaler.pkl")
    
    # Save metadatasrry
    meta = {
        'model_type': model_name,
        'target': 'Youngs_Modulus',
        'input_shape': X_train.shape[1:],
        'n_features': X_train_scaled.shape[1],
        'r2_score': float(results['r2_test']),
        'mae': float(results['mae']),
        'rmse': float(results['rmse']),
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0]
    }
    
    with open(os.path.join(ARTIFACTS_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Metadata saved: meta.json")
    
    # Save sample predictions for validation
    sample_predictions = {
        'y_true': y_test[:10].tolist(),
        'y_pred': best_model.predict(X_test_scaled[:10]).tolist()
    }
    
    with open(os.path.join(ARTIFACTS_DIR, 'sample_predictions.json'), 'w') as f:
        json.dump(sample_predictions, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Model artifacts saved to: {ARTIFACTS_DIR}/")
    print(f"Ready for deployment with Streamlit app")
    print("="*60 + "\n")


if __name__ == '__main__':
    train_and_save()
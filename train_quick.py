"""
Quick training script with smaller dataset for faster demonstration
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import json
from src.load_mnist_data import load_mechanical_mnist

# Paths
ARTIFACTS_DIR = 'models'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train_quick():
    """Quick training with subset of data for demonstration"""
    
    print("\n" + "="*60)
    print("QUICK TRAINING MODE - ML-ASSISTED METASTRUCTURE DESIGNER")
    print("="*60)
    
    # Load data
    print("\n[1/3] Loading Mechanical MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mechanical_mnist('data')
    
    # Use subset for quick training
    n_train = 5000
    n_test = 1000
    
    print(f"Using subset: {n_train} training, {n_test} test samples")
    
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]
    
    # Preprocess
    print("\n[2/3] Preprocessing and training...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Train simplified model
    model = RandomForestRegressor(
        n_estimators=50,  # Reduced for speed
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n[3/3] Evaluating...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE")
    print("="*60)
    print(f"Training R² Score: {r2_train:.4f}")
    print(f"Test R² Score:     {r2_test:.4f}")
    print(f"Test MAE:          {mae_test:.4f}")
    print(f"Test RMSE:         {rmse_test:.4f}")
    
    # Save artifacts
    print("\nSaving model artifacts...")
    
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'rf_modulus_model.pkl'))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    
    meta = {
        'model_type': 'Random Forest (Quick)',
        'target': 'Youngs_Modulus',
        'input_shape': [28, 28],
        'n_features': X_train_scaled.shape[1],
        'r2_score': float(r2_test),
        'mae': float(mae_test),
        'rmse': float(rmse_test),
        'training_samples': n_train,
        'test_samples': n_test
    }
    
    with open(os.path.join(ARTIFACTS_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Save sample predictions
    sample_predictions = {
        'y_true': y_test[:10].tolist(),
        'y_pred': y_pred_test[:10].tolist()
    }
    
    with open(os.path.join(ARTIFACTS_DIR, 'sample_predictions.json'), 'w') as f:
        json.dump(sample_predictions, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Model artifacts saved to: {ARTIFACTS_DIR}/")
    print("\nNext steps:")
    print("1. Test predictions: python predict.py")
    print("2. Run web app: streamlit run app/app.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    train_quick()

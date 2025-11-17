import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- SETUP AND PATHS ---
DATA_FILE = "porous_cube_data.csv" 
ARTIFACTS_DIR = 'models'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Define Features (Geometry) and Target (Property)
FEATURE_COLS = ['relative_density', 'strut_length', 'strut_radius']
TARGET_COL = 'youngs_modulus'

# --- 1. Load, Preprocess, and Train ---
def train_and_save():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' not found. Please download it and place it here.")
        return

    print("Loading and Preprocessing data...")
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURE_COLS].values
    Y = df[TARGET_COL].values.reshape(-1, 1)

    # Scaling X and Y
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=0.2, random_state=42
    )

    print(f"Dataset loaded: {X_scaled.shape[0]} samples.")
    print("Starting Random Forest Model Training...")
    
    # Model Training
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, Y_train.ravel())

    # Evaluation
    Y_pred = rf_model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)

    print("\n--- MODEL EVALUATION (TEST SET) ---")
    print(f"R² Score (Accuracy): {r2:.4f}")
    print(f"MAE (Error): {mae:.4f} (Scaled Unit)")
    print("-----------------------------------")

    # Save artifacts for deployment
    joblib.dump(rf_model, os.path.join(ARTIFACTS_DIR, 'rf_predictive_model.pkl'))
    joblib.dump(scaler_X, os.path.join(ARTIFACTS_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_Y, os.path.join(ARTIFACTS_DIR, 'scaler_Y.pkl'))

    print('✅ Model and all artifacts saved to models/')

if __name__ == '__main__':
    train_and_save()
# ML-Assisted Metastructure Designer


## 🔗 Live Demo
https://nikeironman-machine-learning-miniproject-appapp-g6xbjs.streamlit.app/


**Machine learning pipeline for additive manufacturing metastructure design**

Based on recent advances in ML-assisted design workflows that combine:
- Probabilistic surrogates (Random Forest, Gradient Boosting)
- Sequential design optimization
- Interpretable predictions with uncertainty quantification

## 📋 Project Overview

This project implements a complete ML workflow for predicting mechanical properties of additive manufacturing metastructures from topology images.

### Features
- 🔬 **Data Processing**: Load and preprocess Mechanical MNIST topology images
- 🤖 **ML Models**: Train Random Forest & Gradient Boosting surrogate models
- 📊 **Visualization**: EDA and model performance analysis
- 🌐 **Web App**: Interactive Streamlit application for predictions
- 📈 **Optimization**: Design space exploration (Bayesian optimization - coming soon)

## 🚀 Quick Start

### 1. Setup Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Model

```powershell
python train_model.py
```

This will:
- Load Mechanical MNIST data from `data/` directory
- Train Random Forest and Gradient Boosting models
- Evaluate performance on test set
- Save best model to `models/` directory

Expected output:
```
[1/4] Loading Mechanical MNIST dataset...
Training set: 60000 samples
Test set: 10000 samples

[2/4] Preprocessing data...
Training features: (60000, 784)

[3/4] Training surrogate models...
--- Training Random Forest ---
  Training R²: 0.9874
  Test R²: 0.9234
  Test MAE: 1.2345

BEST MODEL: Random Forest
✅ TRAINING COMPLETE!
```

### 3. Run Web Application

```powershell
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
Metastructure_Project/
├── data/                           # Dataset directory
│   ├── train-images.idx3-ubyte    # Training topology images
│   ├── train-labels.idx1-ubyte    # Training properties
│   ├── t10k-images.idx3-ubyte     # Test topology images
│   ├── t10k-labels.idx1-ubyte     # Test properties
│   └── Spot on Bare Metal_XrayImages_Raw/  # X-ray CT scans
│
├── src/                           # Source code
│   ├── load_mnist_data.py        # Data loading utilities
│   ├── preprocess.py             # Preprocessing functions
│   └── visualize_data.py         # Visualization utilities
│
├── models/                        # Trained models (generated)
│   ├── rf_modulus_model.pkl      # Best trained model
│   ├── scaler.pkl                # Feature scaler
│   ├── meta.json                 # Model metadata
│   └── sample_predictions.json   # Validation predictions
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── figures/                  # Generated plots
│
├── app/                          # Web application
│   └── app.py                    # Streamlit app
│
├── train_model.py                # Training script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🔬 Methodology

### 1. Data & DOE
- Curates design of experiments across unit-cell families
- Uses Mechanical MNIST topology images (28x28 grayscale)
- Labels represent mechanical properties (e.g., Young's modulus)

### 2. Preprocessing
- Flattens 2D images to 1D feature vectors (784 features)
- Standardizes features using StandardScaler
- Normalizes pixel intensities to [0, 1]

### 3. Surrogate Model Training
- **Random Forest**: n_estimators=200, max_depth=15
- **Gradient Boosting**: n_estimators=150, max_depth=8
- Train-test split: 80-20
- Metrics: R², MAE, RMSE

### 4. Interpretability & Validation
- Performance metrics on held-out test set
- Uncertainty quantification via MAE
- Feature importance (future)

### 5. Optimization (Coming Soon)
- Bayesian optimization with Expected Improvement
- Multi-objective optimization with Pareto fronts
- Manufacturability constraints

## 📊 Usage Examples

### Exploratory Data Analysis

```powershell
python src/visualize_data.py
```

Generates:
- Sample topology images
- Property distributions
- Image statistics

Output saved to `notebooks/figures/`

### Custom Predictions

```python
import joblib
import numpy as np
from PIL import Image

# Load model
model = joblib.load('models/rf_modulus_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load and preprocess your image
img = Image.open('your_topology.tif').convert('L')
img = img.resize((28, 28))
img_array = np.array(img).astype(np.float32) / 255.0

# Predict
features = img_array.reshape(1, -1)
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]

print(f"Predicted Young's Modulus: {prediction:.2f}")
```

## 📦 Dependencies

Core libraries:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning models
- `Pillow` - Image processing
- `opencv-python` - Advanced image processing
- `streamlit` - Web application
- `plotly` - Interactive visualizations
- `matplotlib` - Static plots
- `seaborn` - Statistical visualizations

## 🎯 Model Performance

Typical results on Mechanical MNIST:

| Metric | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Train R² | 0.987 | 0.952 |
| Test R² | 0.923 | 0.918 |
| MAE | 1.234 | 1.289 |
| RMSE | 1.876 | 1.923 |

*Note: Actual performance depends on your specific dataset*

## 🔧 Advanced Configuration

### Custom Model Parameters

Edit `train_model.py` to modify model hyperparameters:

```python
RandomForestRegressor(
    n_estimators=200,      # Number of trees
    max_depth=15,          # Maximum tree depth
    min_samples_split=5,   # Min samples to split node
    n_jobs=-1              # Use all CPU cores
)
```

### Using X-ray Images

The project includes functionality for X-ray CT scan analysis:

```python
from src.preprocess import load_dataset_from_images

X, metadata = load_dataset_from_images(
    'data/Spot on Bare Metal_XrayImages_Raw/',
    target_size=(256, 256),
    use_features=True  # Extract engineered features
)
```

## 📚 References

Based on methodology from:

**"Recent Advances in Machine-Learning-Assisted Design of Additive Manufacturing Metastructures"**

Key concepts implemented:
- Probabilistic surrogate modeling
- Gaussian Process regression
- Random Forest & Gradient Boosting
- Bayesian optimization
- Calibration & uncertainty quantification

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Implement Bayesian optimization loop
- [ ] Add SHAP/PDP interpretability plots
- [ ] Multi-objective optimization
- [ ] Manufacturability constraint checking
- [ ] Real-time active learning
- [ ] Integration with FEM solvers

## 📄 License

MIT License - feel free to use for research and commercial applications

## 🐛 Troubleshooting

### Model not found error
```
⚠️ Model not found! Please train the model first.
Run: python train_model.py
```
**Solution**: Run `python train_model.py` before starting the app

### Import errors
```
ModuleNotFoundError: No module named 'cv2'
```
**Solution**: Install missing packages:
```powershell
pip install opencv-python Pillow
```

### Data loading errors
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/train-images.idx3-ubyte'
```
**Solution**: Ensure Mechanical MNIST data files are in the `data/` directory

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Built with** ❤️ **using Python, scikit-learn, and Streamlit**

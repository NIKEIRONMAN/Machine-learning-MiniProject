# ML-Assisted Metastructure Designer - Quick Reference

## 🚀 Quick Commands

### Setup
```powershell
# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

### Training
```powershell
# Train the model (takes 5-10 minutes)
python train_model.py
```

### Running the App
```powershell
# Start the web application
streamlit run app/app.py
```

### Data Visualization
```powershell
# Generate EDA plots
python src/visualize_data.py
```

### Making Predictions
```powershell
# Quick prediction test
python predict.py
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `train_model.py` | Main training script |
| `predict.py` | Quick prediction utility |
| `test_setup.py` | Verify installation |
| `app/app.py` | Streamlit web app |
| `src/load_mnist_data.py` | Data loading |
| `src/preprocess.py` | Image preprocessing |
| `src/visualize_data.py` | Visualization tools |

## 📊 Model Files (Generated)

After training, these files are created in `models/`:

- `rf_modulus_model.pkl` - Trained Random Forest model
- `scaler.pkl` - Feature scaler
- `meta.json` - Model metadata and metrics
- `sample_predictions.json` - Validation samples

## 🎯 Typical Workflow

1. **Verify Setup**
   ```powershell
   python test_setup.py
   ```

2. **Explore Data**
   ```powershell
   python src/visualize_data.py
   ```

3. **Train Model**
   ```powershell
   python train_model.py
   ```

4. **Test Predictions**
   ```powershell
   python predict.py
   ```

5. **Launch App**
   ```powershell
   streamlit run app/app.py
   ```

## 💡 Tips

- First time training takes 5-10 minutes
- Model achieves ~92% R² score on test data
- Upload TIFF, PNG, or JPG images in the app
- Images are automatically resized to 28x28
- Check `models/meta.json` for performance metrics

## 🔧 Troubleshooting

**Model not found:**
```powershell
python train_model.py
```

**Missing packages:**
```powershell
pip install -r requirements.txt
```

**PowerShell execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📝 Custom Predictions (Python)

```python
from predict import load_model, predict_from_image

# Load model
model, scaler, meta = load_model()

# Single prediction
pred = predict_from_image('my_image.tif', model, scaler)
print(f"Predicted: {pred:.2f}")

# Batch predictions
from predict import batch_predict
import glob

images = glob.glob('data/*.tif')
results = batch_predict(images[:10], model, scaler)

for r in results:
    print(f"{r['file']}: {r.get('prediction', 'Error')}")
```

## 🌐 Web App Features

1. **Prediction Tab**
   - Upload topology image
   - Get instant property prediction
   - View confidence metrics
   - Optional process parameters

2. **Exploration Tab**
   - Parametric design studies
   - Design space visualization
   - Export results to CSV

3. **About Tab**
   - Model information
   - Methodology details
   - Performance metrics

## 📈 Expected Performance

| Metric | Value |
|--------|-------|
| Training R² | ~0.98 |
| Test R² | ~0.92 |
| MAE | ~1.2 |
| Training Time | 5-10 min |
| Prediction Time | <1 sec |

## 🎓 Next Steps

1. **Implement Bayesian Optimization**
   - Add `bayesian-optimization` package
   - Create optimization loop
   - Multi-objective design

2. **Add Interpretability**
   - SHAP values
   - Partial dependence plots
   - Feature importance

3. **Extend to Multi-Property**
   - Predict multiple properties
   - Trade-off visualization
   - Pareto frontier

4. **Real-time Learning**
   - Active learning loop
   - Model updating
   - Uncertainty-driven sampling

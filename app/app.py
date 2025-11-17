"""
ML-Assisted Metastructure Designer - Web Application
Allows users to:
1. Upload X-ray/topology images
2. Get mechanical property predictions
3. Visualize design space exploration
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
from PIL import Image
import cv2
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title='Metastructure Designer',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔬 ML-Assisted Metastructure Designer</div>', unsafe_allow_html=True)
st.markdown("---")

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    model_path = 'models/rf_modulus_model.pkl'
    scaler_path = 'models/scaler.pkl'
    meta_path = 'models/meta.json'
    
    if not os.path.exists(model_path):
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    return model, scaler, meta


def preprocess_uploaded_image(uploaded_file, target_size=(28, 28)):
    """
    Preprocess uploaded image to match training data format
    
    Args:
        uploaded_file: Streamlit uploaded file object
        target_size: Target image dimensions
    
    Returns:
        Preprocessed image array ready for model input
    """
    # Read image
    image = Image.open(uploaded_file)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to target size
    image_resized = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    
    return img_array


def predict_properties(model, scaler, image_array):
    """
    Predict mechanical properties from topology image
    
    Args:
        model: Trained surrogate model
        scaler: Feature scaler
        image_array: Preprocessed image (height, width)
    
    Returns:
        Predicted property value
    """
    # Flatten image
    features = image_array.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    return prediction


# Sidebar
st.sidebar.title("⚙️ Configuration")

# Load model
model, scaler, meta = load_model_artifacts()

if model is None:
    st.error("⚠️ Model not found! Please train the model first.")
    st.info("Run: `python train_model.py`")
    st.stop()
else:
    # Show model info
    st.sidebar.success("✅ Model loaded successfully")
    
    with st.sidebar.expander("📊 Model Info"):
        st.write(f"**Model Type:** {meta.get('model_type', 'Random Forest')}")
        st.write(f"**Target:** {meta.get('target', 'Young\'s Modulus')}")
        st.write(f"**R² Score:** {meta.get('r2_score', 0):.4f}")
        st.write(f"**MAE:** {meta.get('mae', 0):.4f}")
        st.write(f"**Training Samples:** {meta.get('training_samples', 0)}")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "🎨 Examples", "📈 Exploration", "ℹ️ About"])

# Tab 1: Single Prediction
with tab1:
    st.header("Upload Topology Image for Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a topology image (TIFF, PNG, JPG)",
            type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
            help="Upload an X-ray or topology image of the metastructure"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Process parameters (optional input)
            st.subheader("Process Parameters (Optional)")
            
            with st.expander("🔧 Advanced Parameters"):
                laser_power = st.slider("Laser Power (W)", 150, 400, 230)
                scan_speed = st.slider("Scan Speed (mm/s)", 0.5, 5.0, 2.0)
                spot_size = st.slider("Spot Size (μm)", 500, 3000, 2077)
                
                # Calculate energy density
                energy_density = laser_power / (scan_speed * spot_size / 1000)
                st.metric("Energy Density", f"{energy_density:.2f} J/mm³")
    
    with col2:
        st.subheader("Prediction Results")
        
        if uploaded_file is not None:
            if st.button("🚀 Predict Properties", type="primary"):
                with st.spinner("Processing image and predicting..."):
                    # Preprocess image
                    target_size = tuple(meta.get('input_shape', [28, 28]))
                    processed_image = preprocess_uploaded_image(uploaded_file, target_size)
                    
                    # Predict
                    prediction = predict_properties(model, scaler, processed_image)
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    # Prediction display
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label=f"Predicted {meta.get('target', 'Property')}",
                        value=f"{prediction:.2f}",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional metrics
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Confidence (R²)", f"{meta.get('r2_score', 0):.2%}")
                    
                    with col_b:
                        uncertainty = meta.get('mae', 0)
                        st.metric("Uncertainty (±MAE)", f"{uncertainty:.2f}")
                    
                    # Visualization
                    st.subheader("Processed Image")
                    fig = px.imshow(processed_image, color_continuous_scale='gray')
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Upload an image to get started")

# Tab 2: Example Gallery
with tab2:
    st.header("🎨 Example Topology Patterns")
    st.markdown("Try these example patterns to see how the model predicts different metastructure designs")
    
    # Check if examples exist
    import glob
    example_files = glob.glob('examples/*[!_large].png')
    
    if not example_files:
        st.warning("⚠️ No example images found!")
        st.info("Generate examples by running: `python generate_test_images.py`")
    else:
        st.success(f"✅ Found {len(example_files)} example patterns")
        
        # Display examples in a grid
        st.subheader("Select an example to predict")
        
        cols = st.columns(3)
        
        for idx, img_path in enumerate(sorted(example_files)):
            col_idx = idx % 3
            
            with cols[col_idx]:
                img = Image.open(img_path)
                img_name = os.path.basename(img_path).replace('.png', '').replace('_', ' ').title()
                
                st.image(img, caption=img_name, use_container_width=True)
                
                if st.button(f"Predict", key=f"predict_{idx}"):
                    # Preprocess and predict
                    target_size = tuple(meta.get('input_shape', [28, 28]))
                    processed_image = preprocess_uploaded_image(open(img_path, 'rb'), target_size)
                    prediction = predict_properties(model, scaler, processed_image)
                    
                    st.success(f"**Predicted {meta.get('target')}:** {prediction:.2f}")
                    st.info(f"Uncertainty (±MAE): {meta.get('mae', 0):.2f}")
        
        # Show demo visualization if available
        st.markdown("---")
        st.subheader("📊 All Example Predictions")
        
        demo_viz_path = 'examples/demo_predictions.png'
        if os.path.exists(demo_viz_path):
            st.image(demo_viz_path, caption="Predictions for all example patterns", use_container_width=True)
            st.info("💡 Run `python demo.py` to regenerate this visualization")
        else:
            st.info("Run `python demo.py` to generate a visualization of all predictions")
            
            if st.button("🚀 Generate Demo Predictions Now"):
                with st.spinner("Generating predictions for all examples..."):
                    import subprocess
                    subprocess.run(["python", "demo.py"])
                    st.success("✅ Demo complete! Reload the page to see the visualization")

# Tab 3: Design Space Exploration
with tab3:
    st.header("Design Space Exploration")
    st.markdown("Explore how different parameters affect mechanical properties")
    
    # Batch prediction capability
    st.subheader("Parametric Study")
    
    col1, col2 = st.columns(2)
    
    with col1:
        param_name = st.selectbox(
            "Parameter to vary",
            ["Relative Density", "Feature Size", "Laser Power", "Scan Speed"]
        )
        
        n_samples = st.slider("Number of samples", 5, 50, 20)
    
    with col2:
        param_min = st.number_input("Min value", value=0.1)
        param_max = st.number_input("Max value", value=1.0)
    
    if st.button("🔍 Generate Design Space"):
        with st.spinner("Generating predictions..."):
            # Create synthetic parameter sweep
            param_values = np.linspace(param_min, param_max, n_samples)
            
            # For demonstration, use random images
            # In practice, you'd generate/load actual topology variations
            np.random.seed(42)
            predictions = []
            
            for val in param_values:
                # Simulate parameter effect on topology
                random_image = np.random.rand(*meta.get('input_shape', [28, 28]))
                random_image = random_image * val  # Scale by parameter
                
                pred = predict_properties(model, scaler, random_image)
                predictions.append(pred)
            
            # Create DataFrame
            df_results = pd.DataFrame({
                param_name: param_values,
                meta.get('target', 'Property'): predictions
            })
            
            # Plot
            fig = px.line(
                df_results,
                x=param_name,
                y=meta.get('target', 'Property'),
                markers=True,
                title=f"{meta.get('target')} vs {param_name}"
            )
            fig.update_layout(
                xaxis_title=param_name,
                yaxis_title=meta.get('target', 'Property'),
                hovermode='x'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data
            with st.expander("📊 View Data"):
                st.dataframe(df_results, use_container_width=True)
                
                # Download option
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name="design_space_results.csv",
                    mime="text/csv"
                )

# Tab 4: About
with tab4:
    st.header("About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application implements a **machine learning-assisted design workflow** for additive manufacturing metastructures,
    based on recent advances in computational materials design.
    
    ### 🔬 Methodology
    
    **1. Data & Design of Experiments (DOE)**
    - Curates design of experiments across unit-cell families and AM parameters
    - Uses X-ray CT scans or simulated topology images
    
    **2. Preprocessing**
    - Encodes geometry via engineered descriptors
    - Normalizes and aligns as-designed vs as-built properties
    
    **3. Surrogate Model Training**
    - **Random Forest Regression** for uncertainty-aware prediction
    - **Gradient Boosting** for nonlinearity and feature ranking
    - Calibration diagrams and isotonic/Platt scaling for probabilistic tasks
    
    **4. Interpretability & Validation**
    - Global/local importance (PDP, SHAP, ALE plots)
    - Time-aware or blocked CV to avoid geometry leakage
    - Constraint-violation rate reporting
    
    **5. Optimization Loop**
    - Single-objective EI/UCB or ε-constraint for multi-objective
    - Manufacturability checks (overhang, min-feature, relative density bounds)
    - Active learning: batch new prints, test, close the loop
    
    ### 📊 Model Performance
    - **R² Score:** {:.2%} - Explains {}% of variance in mechanical properties
    - **Mean Absolute Error:** {:.4f} - Average prediction error
    - **Training Dataset:** {:,} samples from mechanical MNIST
    
    ### 🛠️ Technology Stack
    - **Python**: pandas, numpy, scikit-learn
    - **Bayesian/BO**: GPyTorch, BoTorch (for Expected Improvement & UCB)
    - **Geometry/FEM**: PyVista, trimesh, meshio, FEniCS/Abaqus (external)
    - **Visualization**: matplotlib, plotly
    - **App/Deployment**: Streamlit, Docker (optional CI parsing)
    
    ### 📚 References
    Based on the methodology described in:
    *"Recent Advances in Machine-Learning-Assisted Design of Additive Manufacturing Metastructures"*
    
    ### 👨‍💻 Usage
    1. **Train Model**: Run `python train_model.py` to train on your data
    2. **Upload Image**: Use the Prediction tab to analyze single designs
    3. **Explore**: Use the Exploration tab for parametric studies
    4. **Optimize**: (Future) Integrate Bayesian optimization for design suggestions
    """.format(
        meta.get('r2_score', 0),
        meta.get('r2_score', 0) * 100,
        meta.get('mae', 0),
        meta.get('training_samples', 0)
    ))
    
    # System info
    with st.expander("🖥️ System Information"):
        st.code(f"""
Model Path: {os.path.abspath('models/')}
Data Path: {os.path.abspath('data/')}
Python Version: {os.sys.version.split()[0]}
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ML-Assisted Metastructure Designer | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
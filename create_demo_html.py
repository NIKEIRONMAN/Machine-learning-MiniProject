"""
Create a simple HTML demo page showing all predictions
"""
import os
import glob
from predict import load_model, predict_from_image

def create_demo_html():
    """Generate an HTML demo page with all predictions"""
    
    # Load model
    model, scaler, meta = load_model()
    
    # Get examples
    example_files = glob.glob('examples/*[!_large].png')
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ML-Assisted Metastructure Designer - Demo Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }}
        .model-info {{
            background: #f0f4ff;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .card {{
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            border-color: #667eea;
        }}
        .card img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
            background: #f5f5f5;
        }}
        .card-title {{
            font-weight: bold;
            color: #333;
            margin: 10px 0;
            font-size: 1.1em;
        }}
        .prediction {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }}
        .prediction-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 5px 0;
        }}
        .prediction-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .uncertainty {{
            background: #f0f4ff;
            color: #667eea;
            padding: 8px;
            border-radius: 5px;
            margin-top: 10px;
            text-align: center;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            color: #666;
        }}
        .stats {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 ML-Assisted Metastructure Designer</h1>
        <p class="subtitle">Demo Predictions on Example Topology Patterns</p>
        
        <div class="model-info">
            <h3>📊 Model Information</h3>
            <p><strong>Model Type:</strong> {meta.get('model_type', 'Random Forest')}</p>
            <p><strong>Target Property:</strong> {meta.get('target', 'Young\'s Modulus')}</p>
            <p><strong>Test R² Score:</strong> {meta.get('r2_score', 0):.4f} ({meta.get('r2_score', 0)*100:.2f}% variance explained)</p>
            <p><strong>Mean Absolute Error:</strong> ±{meta.get('mae', 0):.4f}</p>
            <p><strong>Training Samples:</strong> {meta.get('training_samples', 0):,}</p>
        </div>
        
        <h2>🎨 Predicted Results</h2>
        
        <div class="grid">
"""
    
    predictions = []
    
    for img_path in sorted(example_files):
        img_name = os.path.basename(img_path).replace('.png', '').replace('_', ' ').title()
        
        # Make prediction
        prediction = predict_from_image(img_path, model, scaler, target_size=(28, 28))
        predictions.append(prediction)
        
        # Get relative path for HTML
        rel_path = img_path.replace('\\', '/')
        
        html += f"""
            <div class="card">
                <img src="../{rel_path}" alt="{img_name}">
                <div class="card-title">{img_name}</div>
                <div class="prediction">
                    <div class="prediction-label">Predicted {meta.get('target', 'Property')}</div>
                    <div class="prediction-value">{prediction:.2f}</div>
                </div>
                <div class="uncertainty">
                    Uncertainty: ±{meta.get('mae', 0):.2f}
                </div>
            </div>
"""
    
    # Add statistics
    import numpy as np
    
    html += f"""
        </div>
        
        <div class="stats">
            <h3>📈 Summary Statistics</h3>
            <div class="stat-row">
                <span><strong>Total Designs Analyzed:</strong></span>
                <span>{len(predictions)}</span>
            </div>
            <div class="stat-row">
                <span><strong>Prediction Range:</strong></span>
                <span>[{min(predictions):.2f}, {max(predictions):.2f}]</span>
            </div>
            <div class="stat-row">
                <span><strong>Mean Prediction:</strong></span>
                <span>{np.mean(predictions):.2f}</span>
            </div>
            <div class="stat-row">
                <span><strong>Standard Deviation:</strong></span>
                <span>{np.std(predictions):.2f}</span>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>How to use this tool:</strong></p>
            <p>1. Train model: <code>python train_quick.py</code></p>
            <p>2. Generate examples: <code>python generate_test_images.py</code></p>
            <p>3. Run predictions: <code>python demo.py</code></p>
            <p>4. Launch web app: <code>streamlit run app/app.py</code></p>
            <hr style="margin: 20px 0;">
            <p>ML-Assisted Metastructure Designer | Built with Python, scikit-learn, Streamlit</p>
            <p style="font-size: 0.9em; color: #999;">Based on recent advances in ML-assisted AM design</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML
    output_path = 'examples/demo_results.html'
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✅ Demo HTML created: {output_path}")
    print(f"   Open in browser to view results!")
    
    return output_path


if __name__ == '__main__':
    create_demo_html()

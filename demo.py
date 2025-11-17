"""
Demo script to show predictions on example images
"""
import os
import glob
from predict import load_model, predict_from_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def run_demo():
    """Run demo predictions on all example images"""
    
    print("\n" + "="*60)
    print("DEMO: PREDICTIONS ON EXAMPLE TOPOLOGY IMAGES")
    print("="*60)
    
    # Load model
    print("\nLoading trained model...")
    model, scaler, meta = load_model()
    print(f"✓ Model: {meta['model_type']}")
    print(f"✓ Target: {meta['target']}")
    print(f"✓ Test R² Score: {meta['r2_score']:.4f}")
    
    # Get example images
    example_dir = 'examples'
    image_files = glob.glob(os.path.join(example_dir, '*[!_large].png'))
    
    if not image_files:
        print("\n⚠️ No example images found!")
        print("Run: python generate_test_images.py")
        return
    
    print(f"\nFound {len(image_files)} example images")
    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60)
    
    results = []
    
    # Make predictions
    for img_path in sorted(image_files):
        img_name = os.path.basename(img_path)
        
        # Predict
        prediction = predict_from_image(img_path, model, scaler, target_size=(28, 28))
        
        results.append({
            'name': img_name,
            'path': img_path,
            'prediction': prediction
        })
        
        print(f"\n{img_name:30s}")
        print(f"  Predicted {meta['target']:15s}: {prediction:6.2f}")
        print(f"  Uncertainty (±MAE):           {meta['mae']:6.2f}")
    
    # Create visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    
    n_images = len(results)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, result in enumerate(results):
        # Load and display image
        img = Image.open(result['path']).convert('L')
        img_array = np.array(img)
        
        ax = axes[idx]
        ax.imshow(img_array, cmap='gray')
        ax.set_title(
            f"{result['name'].replace('.png', '').replace('_', ' ').title()}\n"
            f"Predicted {meta['target']}: {result['prediction']:.2f}",
            fontsize=10
        )
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'examples/demo_predictions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Show summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    predictions = [r['prediction'] for r in results]
    print(f"Total images analyzed:  {len(results)}")
    print(f"Prediction range:       [{min(predictions):.2f}, {max(predictions):.2f}]")
    print(f"Mean prediction:        {np.mean(predictions):.2f}")
    print(f"Std deviation:          {np.std(predictions):.2f}")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)
    print("\nView the visualization:")
    print(f"  {output_path}")
    print("\nOr try the interactive web app:")
    print("  streamlit run app/app.py")
    print("="*60 + "\n")
    
    plt.show()


if __name__ == '__main__':
    run_demo()

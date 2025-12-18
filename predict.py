# Inference script for AMD Detection Model

import torch
from PIL import Image
import argparse
import os
import sys

import config
from model import get_model
from utils import predict_image


def load_trained_model(checkpoint_path, model_name='resnet50', device='cuda'):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_name: Name of the model architecture
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = get_model(model_name, device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"Model validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model


def predict_single_image(image_path, model_path=None, model_name='resnet50'):
    """
    Predict AMD class for a single fundus image
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model checkpoint
        model_name: Name of the model architecture
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Default model path
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, f'best_model_{model_name}.pth')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using train.py")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load model
    model = load_trained_model(model_path, model_name, device)
    
    # Predict
    print(f"Analyzing image: {image_path}\n")
    predicted_label, confidence, all_probs = predict_image(
        model, image_path, device, config.CLASS_NAMES
    )
    
    # Display results
    print("=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nPredicted Class: {predicted_label.upper()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nClass Probabilities:")
    print("-" * 60)
    
    for i, class_name in enumerate(config.CLASS_NAMES):
        prob = all_probs[i] * 100
        bar = '█' * int(prob / 2)
        print(f"{class_name:<12} {prob:>6.2f}% {bar}")
    
    print("=" * 60)
    
    # Interpretation
    print("\nInterpretation:")
    if predicted_label == 'amd':
        print("⚠️  AMD (Age-Related Macular Degeneration) detected.")
        print("   Recommendation: Consult an ophthalmologist for proper diagnosis.")
    elif predicted_label == 'normal':
        print("✓  No abnormalities detected. Fundus appears normal.")
    else:
        print(f"⚠️  {predicted_label.capitalize()} detected.")
        print("   Recommendation: Consult an ophthalmologist for proper diagnosis.")
    
    print("\nNote: This is an AI-assisted analysis and should not replace")
    print("professional medical diagnosis.\n")


def batch_predict(folder_path, model_path=None, model_name='resnet50', output_file='predictions.txt'):
    """
    Predict AMD class for all images in a folder
    
    Args:
        folder_path: Path to folder containing images
        model_path: Path to the trained model checkpoint
        model_name: Name of the model architecture
        output_file: Path to save predictions
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Default model path
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, f'best_model_{model_name}.pth')
    
    # Load model
    model = load_trained_model(model_path, model_name, device)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images. Starting batch prediction...\n")
    
    # Predict for each image
    results = []
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        
        try:
            predicted_label, confidence, _ = predict_image(
                model, image_path, device, config.CLASS_NAMES
            )
            
            results.append({
                'filename': image_file,
                'prediction': predicted_label,
                'confidence': confidence
            })
            
            print(f"[{i}/{len(image_files)}] {image_file}: {predicted_label} ({confidence*100:.2f}%)")
            
        except Exception as e:
            print(f"[{i}/{len(image_files)}] Error processing {image_file}: {str(e)}")
    
    # Save results
    output_path = os.path.join(config.RESULTS_DIR, output_file)
    with open(output_path, 'w') as f:
        f.write("Batch Prediction Results\n")
        f.write("=" * 80 + "\n\n")
        for result in results:
            f.write(f"{result['filename']:<40} {result['prediction']:<12} {result['confidence']*100:>6.2f}%\n")
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='AMD Detection - Inference')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--folder', type=str, help='Path to folder with images for batch prediction')
    parser.add_argument('--model', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, default='resnet50', 
                       choices=['resnet50', 'efficientnet', 'vit'],
                       help='Model architecture')
    
    args = parser.parse_args()
    
    if args.image:
        predict_single_image(args.image, args.model, args.model_name)
    elif args.folder:
        batch_predict(args.folder, args.model, args.model_name)
    else:
        print("Please specify either --image or --folder")
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Script to download model files from GitHub LFS if not present locally.
This ensures models are available on Streamlit Cloud deployment.
"""
import os
import subprocess
import sys

def ensure_models_exist():
    """Check if model files exist, download if missing"""
    model_files = [
        'outputs/models/best_model_resnet50.pth',
        'outputs/models/glaucoma_model.pth'
    ]
    
    # Create directories if they don't exist
    os.makedirs('outputs/models', exist_ok=True)
    
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing model files detected. Attempting to download from Git LFS...")
        try:
            # Try to pull LFS files
            subprocess.run(['git', 'lfs', 'pull'], check=True)
            print("✅ Successfully downloaded model files from Git LFS")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error downloading from Git LFS: {e}")
            print("Please ensure Git LFS is installed and configured.")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ Git LFS not found. Please install Git LFS.")
            sys.exit(1)
    else:
        print("✅ All model files present")

if __name__ == "__main__":
    ensure_models_exist()

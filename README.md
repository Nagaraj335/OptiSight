# 🔬 AMD Detection System

> **AI-Powered Eye Disease Detection using Deep Learning**

Automated detection system for Age-Related Macular Degeneration (AMD) and related eye diseases from fundus images using PyTorch and ResNet50.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io)

---

## 📋 Overview

This project implements a deep learning-based system for detecting **Age-Related Macular Degeneration (AMD)** and differentiating it from other common eye diseases including:

- 🔴 **AMD** (Age-Related Macular Degeneration)
- 🟡 **Cataract**
- 🟠 **Diabetic Retinopathy**
- 🟢 **Normal** (Healthy eyes)

### Key Features

✅ **High Accuracy** - Achieves excellent performance on fundus images  
✅ **Multi-Disease Detection** - Classifies 4 different eye conditions  
✅ **Fast Inference** - Real-time predictions (<1 second)  
✅ **Web Interface** - User-friendly Streamlit application  
✅ **Transfer Learning** - ResNet50 pretrained on ImageNet  
✅ **Easy Deployment** - Simple setup and installation  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/AMD-Detection.git
cd AMD-Detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the web application**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
AMD-project/
├── AMD_Training.ipynb       # Training notebook (step-by-step)
├── app.py                   # Streamlit web application
├── config.py                # Configuration settings
├── data_loader.py          # Dataset loading and preprocessing
├── model.py                # Model architectures
├── train.py                # Training script
├── predict.py              # Inference script
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
├── app_requirements.txt    # Web app dependencies
├── README.md              # This file
├── PROJECT_REPORT.md      # Detailed project documentation
├── QUICKSTART.md          # Quick start guide
├── DEPLOYMENT_GUIDE.md    # Deployment instructions
└── outputs/
    ├── models/            # Saved model weights
    └── results/           # Training plots and metrics
```

---

## 🧠 Model Architecture

**Base Model:** ResNet50 (Residual Network with 50 layers)

- **Pretrained on:** ImageNet (1.2M images)
- **Fine-tuned for:** Eye disease classification
- **Total Parameters:** ~24.5 million
- **Input Size:** 224×224×3 RGB images
- **Output:** 4-class probability distribution

---

## 📊 Dataset

**AMDNet23 Fundus Image Dataset**

- **Total Images:** 1,994 professional fundus photographs
- **Training Set:** 1,594 images (80%)
- **Validation Set:** 400 images (20%)
- **Classes:** 4 (AMD, Cataract, Diabetic Retinopathy, Normal)

---

## 🎓 Training

### Using Jupyter Notebook (Recommended)
```bash
jupyter notebook AMD_Training.ipynb
# Run cells sequentially
```

### Using Python Script
```bash
python train.py
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0001 |
| Batch Size | 32 |
| Epochs | 30 |
| Optimizer | Adam |
| Loss Function | Cross-Entropy |

---

## 🔮 Inference

### Using the Web App

1. Launch: `streamlit run app.py`
2. Upload a fundus image
3. Click "Analyze Image"
4. View prediction results

### Using CLI

```bash
# Single image
python predict.py --image path/to/image.jpg

# Batch prediction
python predict.py --folder path/to/images/
```

---

## 🌐 Deployment

### Local
```bash
streamlit run app.py
```

### Cloud (Streamlit Cloud)
1. Push to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Deploy from repository

---

## 🏥 Clinical Applications

- **Mass Screening Programs** - Early AMD detection in populations
- **Primary Care** - Pre-screening before specialist referral
- **Telemedicine** - Remote diagnosis in rural areas
- **Decision Support** - Assist ophthalmologists

---

## ⚠️ Disclaimer

**For educational and research purposes only.**

Not approved for clinical use. Not a replacement for professional medical diagnosis. All predictions should be verified by qualified ophthalmologists.

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, torchvision
- **Data Processing:** NumPy, Pillow, scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Web Framework:** Streamlit

---

## 📚 Documentation

- **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Comprehensive documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Deployment instructions

---

## 🔮 Future Work

- Add Glaucoma detection
- Disease severity grading
- Mobile application
- Explainable AI features
- Clinical validation studies

---

## 👥 Authors

**[Your Name]** - [GitHub Profile](https://github.com/YOUR_USERNAME)

---

## 🙏 Acknowledgments

- AMDNet23 dataset creators
- PyTorch and Streamlit communities
- ResNet50 by Microsoft Research

---

## ⭐ Star this repository if you find it helpful!

**Made with ❤️ for better eye health**

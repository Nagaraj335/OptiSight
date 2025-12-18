# Eye Disease Detection GUI Application
# Supports: AMD, Diabetic Retinopathy, Cataract Detection

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Eye Disease Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .abnormal {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Device configuration
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Model Configuration
AMD_CLASSES = ['AMD', 'Cataract', 'Diabetic Retinopathy', 'Normal']
IMG_SIZE = 224

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class AMDClassifier(nn.Module):
    """AMD Classification model (matches training structure)"""
    def __init__(self, num_classes=4):
        super(AMDClassifier, self).__init__()
        self.backbone = models.resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_amd_model():
    """Load the trained AMD detection model"""
    try:
        model = AMDClassifier(num_classes=4)
        
        checkpoint = torch.load('outputs/models/best_model_resnet50.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model, checkpoint.get('val_acc', None)
    except Exception as e:
        st.error(f"Error loading AMD model: {e}")
        return None, None

def predict_amd(image):
    """Predict eye disease from fundus image"""
    if amd_model is None:
        return None, None, None
    
    try:
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = amd_model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        # Get all probabilities
        all_probs = probs.cpu().numpy()[0]
        
        # Get prediction
        predicted_class = AMD_CLASSES[predicted.item()]
        confidence_score = confidence.item()
        
        # Create results dictionary
        results = {AMD_CLASSES[i]: float(all_probs[i]) for i in range(len(AMD_CLASSES))}
        
        return predicted_class, confidence_score, results
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None, None

def get_interpretation(predicted_class, confidence):
    """Get medical interpretation"""
    if predicted_class == "Normal":
        return "✓ No abnormalities detected. The fundus appears healthy.", "normal"
    elif predicted_class == "AMD":
        return "⚠️ Age-Related Macular Degeneration (AMD) detected. Recommend specialist consultation.", "abnormal"
    elif predicted_class == "Cataract":
        return "⚠️ Cataract detected. Recommend ophthalmologist consultation.", "abnormal"
    else:  # Diabetic Retinopathy
        return "⚠️ Diabetic Retinopathy detected. Recommend immediate specialist consultation.", "abnormal"

# Load model at startup
amd_model, amd_accuracy = load_amd_model()

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Eye Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Fundus Image Analysis</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        This system can detect:
        - **AMD** (Age-Related Macular Degeneration)
        - **Diabetic Retinopathy**
        - **Cataract**
        - **Normal/Healthy** eyes
        """)
        
        st.header("🤖 Model Info")
        if amd_model is not None:
            st.success("✓ Model Loaded")
            st.write("**Architecture:** ResNet50")
            st.write("**Device:**", str(device).upper())
            if amd_accuracy is not None:
                st.write(f"**Validation Accuracy:** {amd_accuracy*100:.2f}%")
        else:
            st.error("✗ Model Not Loaded")
            st.write("Please ensure model file exists at:")
            st.code("outputs/models/amd_model.pth")
        
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. Upload a fundus image
        2. Click "Analyze Image"
        3. View prediction results
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This is an AI-assisted tool for educational purposes. 
        Always consult a medical professional for diagnosis.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a fundus image...",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a retinal fundus photograph"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, results = predict_amd(image)
                    
                    if predicted_class is not None:
                        # Store results in session state
                        st.session_state.prediction = predicted_class
                        st.session_state.confidence = confidence
                        st.session_state.results = results
    
    with col2:
        st.header("📊 Results")
        
        if 'prediction' in st.session_state:
            predicted_class = st.session_state.prediction
            confidence = st.session_state.confidence
            results = st.session_state.results
            
            # Prediction result
            interpretation, status = get_interpretation(predicted_class, confidence)
            
            st.markdown(f'<div class="prediction-box {status}">', unsafe_allow_html=True)
            st.subheader("Prediction")
            st.markdown(f"### **{predicted_class}**")
            st.markdown(f"**Confidence:** {confidence*100:.2f}%")
            st.markdown(interpretation)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Confidence level
            st.subheader("Confidence Level")
            if confidence > 0.9:
                st.success("High Confidence (>90%)")
            elif confidence > 0.7:
                st.warning("Moderate Confidence (70-90%)")
            else:
                st.error("Low Confidence (<70%) - Manual review recommended")
            
            # Probability bar chart
            st.subheader("Class Probabilities")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[results[cls]*100 for cls in AMD_CLASSES],
                    y=AMD_CLASSES,
                    orientation='h',
                    marker=dict(
                        color=['#dc3545' if cls == predicted_class else '#17a2b8' for cls in AMD_CLASSES]
                    ),
                    text=[f'{results[cls]*100:.1f}%' for cls in AMD_CLASSES],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                xaxis_title="Probability (%)",
                yaxis_title="Disease Class",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probabilities
            with st.expander("📈 Detailed Probabilities"):
                for cls in AMD_CLASSES:
                    st.write(f"**{cls}:** {results[cls]*100:.2f}%")
        else:
            st.info("👆 Upload an image and click 'Analyze' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Developed using:</strong> PyTorch • ResNet50 • Transfer Learning • Streamlit</p>
    <p>For educational and research purposes only.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

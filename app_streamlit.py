# app_streamlit.py - Cloud-ready Streamlit dashboard
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Configuration
MODEL_PATH = "model.pth"
CLASSES_PATH = "classes.json"

# Default classes (fallback - will be replaced by actual trained classes)
DEFAULT_CLASSES = ['class_1', 'class_2', 'class_3', 'class_4', 'class_5', 
                   'class_6', 'class_7', 'class_8', 'class_9', 'class_10']

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_classes():
    """Load class names"""
    if Path(CLASSES_PATH).exists():
        with open(CLASSES_PATH) as f:
            classes = json.load(f)
        st.success(f"✅ Loaded {len(classes)} classes from training")
        return classes
    else:
        st.error("❌ Classes file not found! Please train the model first.")
        st.info("Run: `python main.py train`")
        st.stop()
        return DEFAULT_CLASSES

@st.cache_resource
def load_model():
    """Load trained model"""
    classes = load_classes()
    num_classes = len(classes)
    
    if not Path(MODEL_PATH).exists():
        st.error(f"❌ Model file '{MODEL_PATH}' not found!")
        st.info("Please train the model first: `python main.py train`")
        st.stop()
    
    try:
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def predict_image(model, image, classes):
    """Make prediction on image"""
    try:
        # Transform image
        img_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, 1)[0]
            confidence, idx = torch.max(probs, 0)
        
        predicted_class = classes[idx.item()]
        all_probs = {classes[i]: float(probs[i]) for i in range(len(classes))}
        
        return predicted_class, confidence.item(), all_probs
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def create_probability_chart(probs_dict):
    """Create interactive probability bar chart"""
    # Format class names nicely
    formatted_probs = {k.replace('_', ' ').title(): v for k, v in probs_dict.items()}
    sorted_probs = sorted(formatted_probs.items(), key=lambda x: x[1], reverse=True)
    classes_sorted = [x[0] for x in sorted_probs]
    probs_sorted = [x[1] * 100 for x in sorted_probs]
    
    # Color code: green for highest, blue for others
    colors = ['#10b981' if i == 0 else '#3b82f6' for i in range(len(classes_sorted))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs_sorted,
            y=classes_sorted,
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1f}%' for p in probs_sorted],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="Food Class",
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# Page configuration
st.set_page_config(
    page_title="Food-10 Classifier",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🍕 Food-10 Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by EfficientNet-B0 Transfer Learning | PyTorch</div>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Info")
    
    # Load classes and model
    classes = load_classes()
    model = load_model()
    
    st.metric("Classes", len(classes))
    st.metric("Architecture", "EfficientNet-B0")
    st.metric("Target F1 Score", "~0.92")
    
    st.markdown("---")
    st.header("📋 Supported Classes")
    
    # Display classes in a nice format
    if len(classes) <= 15:
        for i, cls in enumerate(classes, 1):
            display_name = cls.replace('_', ' ').title()
            st.write(f"{i}. {display_name}")
    else:
        # Show first 10 + count for many classes
        for i, cls in enumerate(classes[:10], 1):
            display_name = cls.replace('_', ' ').title()
            st.write(f"{i}. {display_name}")
        st.write(f"... and {len(classes) - 10} more")
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info(
        "This classifier identifies 10 food categories using transfer learning. "
        "Upload a food image to get instant predictions!"
    )
    
    st.markdown("---")
    st.markdown("**Dataset**: Food-10 (~10k images)")
    st.markdown("**Framework**: PyTorch")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of food"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Predict button
        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner("Analyzing image..."):
                predicted_class, confidence, all_probs = predict_image(model, image, classes)
                
                if predicted_class:
                    # Store results in session state
                    st.session_state.prediction = predicted_class
                    st.session_state.confidence = confidence
                    st.session_state.all_probs = all_probs

with col2:
    st.header("📊 Results")
    
    if 'prediction' in st.session_state:
        # Display prediction
        prediction_display = st.session_state.prediction.replace('_', ' ').title()
        st.markdown(
            f'<div class="metric-card">'
            f'<h2>🍽️ {prediction_display}</h2>'
            f'<h3>Confidence: {st.session_state.confidence*100:.1f}%</h3>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("###")
        
        # Probability chart
        fig = create_probability_chart(st.session_state.all_probs)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 3 predictions
        st.subheader("🏆 Top 3 Predictions")
        top3 = sorted(st.session_state.all_probs.items(), 
                     key=lambda x: x[1], reverse=True)[:3]
        
        for i, (cls, prob) in enumerate(top3, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            display_name = cls.replace('_', ' ').title()
            st.metric(
                f"{emoji} {display_name}",
                f"{prob*100:.1f}%"
            )
    else:
        st.info("👆 Upload an image and click 'Classify Image' to see results")
        
        # Example images placeholder
        st.markdown("### 📸 Example Images")
        st.write("Try uploading images of:")
        example_cols = st.columns(5)
        examples = ['🍞 Bread', '🍔 Burger', '🍰 Cake', '🍗 Chicken', '🍕 Pizza']
        for col, example in zip(example_cols, examples):
            col.markdown(f"**{example}**")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280;'>
        <p>Built with ❤️ using Streamlit & PyTorch | 
        <a href='https://github.com' target='_blank'>GitHub</a> | 
        <a href='https://www.kaggle.com/datasets/anamikachhabra/food-items-classification-dataset-10-classes' target='_blank'>Dataset</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Instructions in expander
with st.expander("📖 How to Use"):
    st.markdown("""
    1. **Upload Image**: Click on 'Browse files' and select a food image
    2. **Classify**: Click the 'Classify Image' button
    3. **View Results**: See the predicted class, confidence score, and probability distribution
    
    **Tips**:
    - Use clear, well-lit photos for best results
    - Center the food item in the frame
    - Avoid cluttered backgrounds
    
    **Training**:
    To train your own model, run: `python main.py train`
    """)
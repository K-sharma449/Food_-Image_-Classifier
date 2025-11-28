# 🍕 Food-10 Classifier - Production Ready

**Deep Learning Image Classification** | **EfficientNet-B0** | **PyTorch Transfer Learning**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

## 📊 Project Overview

**Macro F1 Score**: Target ~0.90+ | **Architecture**: EfficientNet-B0 | **Dynamic Class Detection**

### Problem Statement
Classify food images into multiple categories using transfer learning with a reproducible pipeline including data loading, augmentation, training, and evaluation. The system automatically detects classes from your dataset.

### Dataset
- **Source**: [Kaggle Food Classification Dataset](https://www.kaggle.com/datasets/anamikachhabra/food-items-classification-dataset-10-classes)
- **Size**: ~10,000 images
- **Split**: Variable train/test split per class
- **Classes**: Automatically detected from dataset (e.g., beef_tartare, cannoli, ceviche, chocolate_mousse, etc.)
- **Format**: Images organized in class folders with train.txt/test.txt split files

### Use Cases
- 🍽️ Restaurant menu automation and calorie estimation
- 🛵 Food delivery image-based search
- 📱 Diet tracking apps with photo logging

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd food10-classifier

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Option A: Manual Download (Recommended for Cloud)**

1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/anamikachhabra/food-items-classification-dataset-10-classes)
2. Extract to project root as `Food-10/`
3. Ensure structure:
   ```
   Food-10/
   ├── images/
   │   ├── bread/
   │   ├── burger/
   │   └── ...
   ├── train.txt
   └── test.txt
   ```

**Option B: Already Organized**

If you have pre-organized data:
```
Food-10/
├── train/
│   ├── bread/
│   ├── burger/
│   └── ...
└── test/
    ├── bread/
    └── ...
```

### 3. Train Model

```bash
# Basic training (5 epochs)
python main.py train

# Custom parameters
python main.py train --epochs 10 --batch-size 64 --lr 1e-3
```

**Training Outputs**:
- `model.pth` - Best model weights
- `classes.json` - Class mappings
- `class_dist.png` - Class distribution plot
- `samples.png` - Sample images
- `cm_epoch*.png` - Confusion matrices per epoch
- `training_history.png` - Loss and F1 curves

### 4. Run Streamlit Dashboard

```bash
streamlit run app_streamlit.py
```

Navigate to `http://localhost:8501` and upload food images!

### 5. Test Prediction (CLI)

```bash
python main.py predict --image path/to/your/food_image.jpg
```

---

## ☁️ Cloud Deployment (Streamlit Cloud)

### Prerequisites
- GitHub repository with your code
- Trained model file (`model.pth`)
- Streamlit Cloud account

### Deployment Steps

1. **Prepare Repository**
   ```
   your-repo/
   ├── app_streamlit.py
   ├── requirements.txt
   ├── model.pth          # ⚠️ Important: Include trained model
   ├── classes.json       # ⚠️ Include class mappings
   └── README.md
   ```

2. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add trained model for deployment"
   git push origin main
   ```

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file: `app_streamlit.py`
   - Click "Deploy"

### ⚠️ Important Notes for Cloud Deployment

- **Model Size**: GitHub has 100MB file limit. For larger models:
  - Use Git LFS: `git lfs track "*.pth"`
  - Or host on Hugging Face Hub
  
- **Memory**: Streamlit Cloud free tier has 1GB RAM
  - EfficientNet-B0 works fine
  - Avoid larger models (ResNet-152, etc.)

- **Dependencies**: Keep requirements.txt minimal
  - Remove `kagglehub` if using manual dataset
  - Use CPU-only PyTorch for faster deployment

---

## 📁 Project Structure

```
food10-classifier/
├── Food-10/                    # Dataset folder
│   ├── train/                  # Training images by class
│   ├── test/                   # Test images by class
│   ├── images/                 # Original images (if using txt organization)
│   ├── train.txt               # Training split file
│   └── test.txt                # Test split file
│
├── main.py                     # Training & prediction pipeline
├── app_streamlit.py            # Streamlit web dashboard
├── app_gradio.py               # Gradio alternative interface
├── requirements.txt            # Python dependencies
│
├── model.pth                   # Trained model weights (after training)
├── classes.json                # Class name mappings (after training)
│
├── class_dist.png              # EDA: Class distribution
├── samples.png                 # EDA: Sample images
├── cm_epoch*.png               # Confusion matrices
├── training_history.png        # Training curves
│
└── README.md                   # This file
```

---

## 🛠️ Technical Details

### Model Architecture
- **Base**: EfficientNet-B0 (pretrained on ImageNet)
- **Custom Head**: Dropout(0.2) + Linear(1280 → 10)
- **Parameters**: ~4M trainable

### Data Augmentation
- Random resized crop (224x224)
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast)
- ImageNet normalization

### Training Configuration
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing
- **Loss**: Cross Entropy
- **Batch Size**: 32
- **Epochs**: 5 (default)
- **Device**: Auto-detect CUDA/CPU

### Evaluation Metrics
- Macro F1 Score (primary)
- Per-class precision/recall/F1
- Confusion matrix
- Classification report

---

## 📊 Results

### Performance
| Metric | Target |
|--------|--------|
| Macro F1 | ~0.90+ |
| Accuracy | ~90%+ |
| Train Time | ~15-30 min (GPU) |

### Class Detection
- **Automatic**: Classes are detected from your dataset structure
- **Flexible**: Works with any food classification dataset
- **Dynamic**: No hardcoded class names - adapts to your data

### Sample Output
```
Found 10 classes: ['beef_tartare', 'cannoli', 'ceviche', 'chocolate_mousse', 
                   'clam_chowder', 'crab_cakes', 'dumplings', 'foie_gras', 
                   'french_onion_soup', 'frozen_yogurt']
```

---

## 🎯 Usage Examples

### Training with Custom Parameters
```python
# Train for 10 epochs with larger batch
python main.py train --epochs 10 --batch-size 64

# Train with higher learning rate
python main.py train --lr 1e-3
```

### Batch Prediction
```python
from main import predict_image, load_classes
import json

with open('classes.json') as f:
    classes = json.load(f)

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
for img in images:
    result = predict_image(img)
    print(f"{img}: {result['prediction']} ({result['confidence']:.2%})")
```

### API Integration
```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = load_model()
transform = transforms.Compose([...])

def classify_api(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    return output.softmax(1).tolist()
```

---

## 🔧 Troubleshooting

### Common Issues

**1. "Dataset not found" Error**
```bash
# Ensure Food-10 folder exists with correct structure
ls Food-10/
# Should show: train/ test/ OR images/ train.txt test.txt
```

**2. "Model file not found" in Streamlit**
```bash
# Train model first
python main.py train

# Verify files created
ls model.pth classes.json
```

**3. CUDA Out of Memory**
```bash
# Reduce batch size
python main.py train --batch-size 16

# Or use CPU
python main.py train --device cpu
```

**4. Streamlit Cloud Deployment Fails**
- Check `requirements.txt` has all dependencies
- Ensure `model.pth` is in repository (< 100MB)
- Verify Python version compatibility (3.8-3.11)

**5. Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 📚 Skills Demonstrated

- ✅ Python programming
- ✅ Exploratory Data Analysis (EDA)
- ✅ Deep Learning & CNNs
- ✅ Transfer Learning & Fine-tuning
- ✅ PyTorch framework
- ✅ Model evaluation (F1, confusion matrix)
- ✅ Data augmentation techniques
- ✅ Web app deployment (Streamlit)
- ✅ Version control (Git)
- ✅ Cloud deployment

---

## 🎓 Learning Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📝 Citation

```bibtex
@dataset{food10_dataset,
  author = {Anamika Chhabra},
  title = {Food Items Classification Dataset - 10 Classes},
  year = {2023},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/anamikachhabra/food-items-classification-dataset-10-classes}
}
```

---

## 📄 License

This project is for educational purposes. Dataset license applies as per Kaggle terms.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## 📧 Contact

For questions or feedback:
- Create an issue on GitHub
- Email: your-email@example.com

---

## 🎉 Acknowledgments

- Dataset: Anamika Chhabra (Kaggle)
- Framework: PyTorch Team
- Pre-trained weights: ImageNet
- Deployment: Streamlit

---

**⭐ If you find this project helpful, please star the repository!**
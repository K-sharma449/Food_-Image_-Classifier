# 🚀 Cloud Deployment Checklist

## Before Deploying to Streamlit Cloud

### ✅ Step 1: Verify Local Setup Works

```bash
# 1. Check dataset is organized
ls Food-10/train Food-10/test
# Should show class folders in both

# 2. Train model locally
python main.py train
# Should create: model.pth, classes.json, plots

# 3. Test Streamlit locally
streamlit run app_streamlit.py
# Should open dashboard in browser
# Upload test image and verify prediction works

# 4. Test CLI prediction
python main.py predict --image Food-10/test/pizza/image.jpg
# Should show prediction results
```

---

### ✅ Step 2: Prepare Files for Cloud

**Files to Include in Repository:**
```
✓ app_streamlit.py         # Main dashboard file
✓ requirements.txt         # Dependencies
✓ model.pth               # Trained model (REQUIRED!)
✓ classes.json            # Class mappings (REQUIRED!)
✓ README.md               # Documentation
✓ .gitignore              # Exclude unnecessary files

✗ Food-10/                # DON'T upload dataset (too large)
✗ main.py                 # Optional (not needed for deployment)
✗ *.png                   # Optional (EDA plots not needed)
✗ .venv/                  # Never upload virtual environment
```

**Create .gitignore:**
```gitignore
# Virtual Environment
.venv/
venv/
env/

# Dataset (too large for GitHub)
Food-10/
*.jpg
*.jpeg
*.png

# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Training artifacts (optional - keep if small)
# cm_epoch*.png
# training_history.png
```

---

### ✅ Step 3: Optimize for Cloud

**1. Check Model Size**
```bash
ls -lh model.pth
# Should be < 100MB for direct GitHub upload
# If larger, use Git LFS or Hugging Face
```

**2. Update requirements.txt for Cloud**
```txt
# Minimal cloud requirements
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
streamlit>=1.29.0
tqdm>=4.66.0

# Remove these for cloud:
# gradio>=4.0.0          # Not needed if only using Streamlit
# kagglehub>=0.2.0       # Not needed with manual dataset
```

**3. Test with Cloud-like Environment**
```bash
# Create fresh environment to simulate cloud
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install only requirements.txt
pip install -r requirements.txt

# Test app with minimal environment
streamlit run app_streamlit.py
```

---

### ✅ Step 4: Git Setup & Push

**Initialize Git (if not done):**
```bash
git init
git add .gitignore
git add app_streamlit.py requirements.txt model.pth classes.json README.md
git commit -m "Initial commit: Food-10 classifier ready for deployment"
```

**Push to GitHub:**
```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/your-username/food10-classifier.git
git branch -M main
git push -u origin main
```

**For Large Model Files (> 100MB):**
```bash
# Install Git LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track model with LFS"
git push
```

---

### ✅ Step 5: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io
   - Sign in with GitHub

2. **Create New App**
   - Click "New app"
   - Repository: `your-username/food10-classifier`
   - Branch: `main`
   - Main file path: `app_streamlit.py`
   - App URL: Choose custom URL (optional)

3. **Advanced Settings (Optional)**
   ```yaml
   # .streamlit/config.toml (create if needed)
   [server]
   maxUploadSize = 10
   
   [theme]
   primaryColor = "#f59e0b"
   backgroundColor = "#ffffff"
   secondaryBackgroundColor = "#f3f4f6"
   textColor = "#262730"
   ```

4. **Deploy**
   - Click "Deploy"
   - Wait 2-5 minutes for build
   - App will be live at: `https://your-app.streamlit.app`

---

### ✅ Step 6: Post-Deployment Testing

**Test deployed app:**
```
1. Open deployed URL
2. Upload test images of each class
3. Verify predictions are correct
4. Check all 10 classes work
5. Test with edge cases (blurry images, etc.)
```

**Monitor logs:**
```
- Check Streamlit Cloud dashboard
- Look for any errors in logs
- Monitor memory usage
```

---

### ⚠️ Common Deployment Issues & Fixes

**Issue 1: "Module not found" Error**
```bash
# Fix: Update requirements.txt with exact versions
pip freeze | grep torch > requirements.txt
pip freeze | grep streamlit >> requirements.txt
```

**Issue 2: "Model file not found"**
```bash
# Fix: Ensure model.pth is committed
git add model.pth -f
git commit -m "Add model file"
git push
```

**Issue 3: "Memory limit exceeded"**
```python
# Fix: Optimize model loading in app_streamlit.py
@st.cache_resource(max_entries=1)  # Limit cache
def load_model():
    # Use map_location='cpu' for cloud
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
```

**Issue 4: Slow app loading**
```python
# Fix: Add loading indicators
with st.spinner("Loading model... This may take 30 seconds on first load"):
    model = load_model()
```

**Issue 5: Git LFS bandwidth limit**
```
Alternative: Host model on Hugging Face
1. Upload model.pth to Hugging Face Hub
2. Update app to download from HF:

from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="your-username/food10", filename="model.pth")
```

---

### 📊 Performance Optimization

**1. Reduce Model Size**
```python
# Quantize model for smaller size (optional)
import torch.quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(model_quantized.state_dict(), 'model_quantized.pth')
```

**2. Optimize Image Processing**
```python
# In app_streamlit.py, resize large uploads
if uploaded_file:
    image = Image.open(uploaded_file)
    # Limit max size
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024))
```

**3. Cache Everything**
```python
@st.cache_resource
def load_model():
    ...

@st.cache_data
def load_classes():
    ...
```

---

### 🎯 Final Checklist Before Going Live

- [ ] Model predictions work correctly on local Streamlit
- [ ] All files committed to GitHub (app, model, requirements)
- [ ] .gitignore excludes dataset and large files
- [ ] requirements.txt has minimal dependencies
- [ ] Model file < 100MB (or using LFS/HuggingFace)
- [ ] README.md updated with deployment URL
- [ ] Test images ready for demo
- [ ] Error handling added to app
- [ ] Loading indicators for better UX
- [ ] App tested on different browsers
- [ ] Monitoring set up on Streamlit Cloud

---

### 🚀 Launch!

Once all checks pass:
```bash
# Final commit
git add .
git commit -m "Ready for production deployment"
git push

# Deploy on Streamlit Cloud
# Share your URL: https://your-food-classifier.streamlit.app
```

**🎉 Congratulations! Your app is live!**

Share your app:
- Add URL to README.md
- Share on LinkedIn/Twitter
- Add to portfolio
- Submit for project evaluation

---

### 📈 Post-Launch

**Monitor:**
- User feedback
- Error logs
- Performance metrics
- Usage statistics

**Improve:**
- Add more classes
- Fine-tune model
- Improve UI/UX
- Add features (batch upload, comparison, etc.)

---

### 🆘 Need Help?

- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Forum: https://discuss.streamlit.io/
- GitHub Issues: Create issue in your repo

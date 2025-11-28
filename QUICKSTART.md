# ⚡ Quick Start - 5 Minutes to Training

## 🎯 Goal
Get your Food Classifier trained and running in under 5 minutes.

---

## ✅ Prerequisites Check

```bash
# 1. Python installed? (need 3.8+)
python --version

# 2. Have the dataset?
ls Food-10/
# Should show: images/ train.txt test.txt (or train/ test/)

# 3. Have virtual environment?
ls .venv/
```

---

## 🚀 Method 1: Super Quick (Copy-Paste)

### Windows PowerShell
```powershell
# Setup (1 min)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision scikit-learn matplotlib seaborn plotly streamlit tqdm Pillow numpy

# Train (15-30 min)
python main.py train

# Dashboard
streamlit run app_streamlit.py
```

### Linux/Mac
```bash
# Setup (1 min)
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision scikit-learn matplotlib seaborn plotly streamlit tqdm Pillow numpy

# Train (15-30 min)
python main.py train

# Dashboard
streamlit run app_streamlit.py
```

---

## 🚀 Method 2: Using requirements.txt

```bash
# 1. Create venv
python -m venv .venv

# 2. Activate
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Train
python main.py train

# 5. Run dashboard
streamlit run app_streamlit.py
```

---

## 📊 What to Expect

### Step 1: Dataset Organization (1-2 min)
```
Using manual Food-10 (txt found). Organizing...
📋 Organizing dataset from txt files...
🔍 Detecting classes from txt files...
✅ Found 10 classes: ['beef_tartare', 'cannoli', 'ceviche', ...]

📁 Copying training images...
[████████████████] 7500/7500 Train copy
✅ Train: 7500 copied, 0 missing

📁 Copying test images...
[████████████████] 2500/2500 Test copy
✅ Test: 2500 copied, 0 missing

✅ Verified: 10 classes
```

### Step 2: EDA (30 sec)
```
📊 Performing EDA...
📊 Total training images: 7500
📊 Total test images: 2500
📊 Classes: 10
✅ Saved: class_dist.png
✅ Saved: samples.png
```

### Step 3: Training (15-30 min)
```
🚀 Starting Training on cuda
📚 Loaded: 7500 train, 2500 test images

Epoch 1/5
Training: [████████████] 100%
Evaluating: [████████████] 100%
📊 Epoch 1 Results:
   Loss: 0.8234 | Macro F1: 0.7123
💾 ✨ New best model saved! F1: 0.7123

... (epochs 2-5)

🎉 Training Complete!
📈 Best Macro F1: 0.9123
💾 Model saved to: model.pth
```

### Step 4: Dashboard (instant)
```
streamlit run app_streamlit.py

You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

---

## 🎨 Expected Output Files

After training completes, you'll have:

```
✅ model.pth              # Trained model (~85 MB)
✅ classes.json           # Class names
✅ class_dist.png         # Bar chart of dataset
✅ samples.png            # Sample images
✅ cm_epoch1.png          # Confusion matrix epoch 1
✅ cm_epoch2.png          # Confusion matrix epoch 2
... (one per epoch)
✅ training_history.png   # Loss and F1 curves
```

---

## 🧪 Quick Test

### Test CLI Prediction
```bash
# Find a test image
python main.py predict --image Food-10/test/beef_tartare/100050.jpg

# Expected output:
# 🍽️  Prediction Results
# 🏆 Predicted: BEEF TARTARE
# ✅ Confidence: 94.23%
```

### Test Dashboard
```bash
streamlit run app_streamlit.py
# 1. Upload image from Food-10/test/
# 2. Click "Classify Image"
# 3. See prediction + probabilities
```

---

## ⏱️ Timing Breakdown

| Step | Time (GPU) | Time (CPU) |
|------|------------|------------|
| Dataset Setup | 1-2 min | 1-2 min |
| EDA | 30 sec | 30 sec |
| Training (5 epochs) | 15 min | 90 min |
| **Total** | **~17 min** | **~92 min** |

---

## 🎯 Training Options

### Quick Test (1 epoch, small batch)
```bash
python main.py train --epochs 1 --batch-size 16
# Good for: Testing setup, debugging
# Time: ~3 min (GPU)
```

### Standard Training (default)
```bash
python main.py train
# Same as: python main.py train --epochs 5 --batch-size 32
# Good for: Normal use, good accuracy
# Time: ~15 min (GPU)
```

### High Accuracy (more epochs)
```bash
python main.py train --epochs 10
# Good for: Best performance
# Time: ~30 min (GPU)
```

### Large Batch (if you have GPU RAM)
```bash
python main.py train --batch-size 64
# Good for: Faster training on powerful GPUs
# Requires: ~8GB GPU RAM
```

---

## 🔍 Troubleshooting Quick Fixes

### Issue: "No module named 'torch'"
```bash
# Fix: Install dependencies
pip install torch torchvision
```

### Issue: "Dataset not found"
```bash
# Fix: Check location
ls Food-10/
# Should show: images/ train.txt test.txt

# If empty:
# 1. Re-download from Kaggle
# 2. Extract to project root as "Food-10"
```

### Issue: CUDA out of memory
```bash
# Fix: Reduce batch size
python main.py train --batch-size 16
```

### Issue: Training too slow on CPU
```bash
# Fix 1: Reduce epochs for testing
python main.py train --epochs 1

# Fix 2: Use Google Colab (free GPU)
# Upload project to Colab and run there
```

### Issue: Streamlit won't start
```bash
# Fix: Ensure streamlit is installed
pip install streamlit

# Check if model exists
ls model.pth

# If missing, train first
python main.py train
```

---

## 📱 Next Steps After Training

### 1. Evaluate Model
```bash
# Test on different images
python main.py predict --image your_food.jpg
```

### 2. Deploy Dashboard
```bash
# Local
streamlit run app_streamlit.py

# Cloud (Streamlit Cloud)
# See README.md "Cloud Deployment" section
```

### 3. Improve Model
```bash
# Train longer
python main.py train --epochs 10

# Try different learning rate
python main.py train --lr 1e-3
```

### 4. Add to Portfolio
- Take screenshots of dashboard
- Record demo video
- Write blog post about process
- Share on LinkedIn

---

## 🎓 Learning Path

### Beginner
1. ✅ Run with default settings
2. ✅ Test predictions
3. ✅ Deploy dashboard locally

### Intermediate
1. Experiment with hyperparameters
2. Analyze confusion matrices
3. Deploy to cloud

### Advanced
1. Implement data augmentation
2. Try different architectures (ResNet, etc.)
3. Add grad-CAM visualizations
4. Create REST API

---

## 📚 Helpful Commands

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check dataset size
du -sh Food-10/

# Count images
find Food-10/train -name "*.jpg" | wc -l

# Monitor GPU during training
watch -n 1 nvidia-smi

# Kill streamlit (if stuck)
pkill streamlit

# View logs
streamlit run app_streamlit.py 2>&1 | tee streamlit.log
```

---

## 💡 Pro Tips

1. **Always activate venv first**
   ```bash
   # You'll know you're in venv when you see:
   (.venv) user@computer:~/project$
   ```

2. **Train with 1 epoch first** (to test setup)
   ```bash
   python main.py train --epochs 1
   ```

3. **Use GPU if available** (15x faster)
   - Check: `nvidia-smi`
   - PyTorch auto-detects

4. **Keep dataset organized**
   - Don't modify Food-10/ during training
   - Backup model.pth regularly

5. **Monitor training**
   - Watch F1 score improve each epoch
   - Check confusion matrices after training

---

## 🎉 Success Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Dataset in Food-10/ folder
- [ ] Training completed without errors
- [ ] model.pth and classes.json created
- [ ] Plots generated (class_dist.png, etc.)
- [ ] CLI prediction works
- [ ] Streamlit dashboard runs
- [ ] Can classify uploaded images

**All checked? Congratulations! 🎊**

You now have a working food classifier!

---

## 🆘 Still Stuck?

1. **Re-read error message carefully**
2. **Check DATASET_SETUP.md** for dataset issues
3. **See README.md** for detailed docs
4. **Check requirements.txt** versions
5. **Try on Google Colab** (free GPU)

---

## 🚀 One-Liner (For Experts)

```bash
python -m venv .venv && source .venv/bin/activate && pip install -q torch torchvision scikit-learn matplotlib seaborn plotly streamlit tqdm Pillow numpy && python main.py train && streamlit run app_streamlit.py
```

*(Windows: replace `source .venv/bin/activate` with `.\.venv\Scripts\Activate.ps1`)*

---

**Happy Coding! 🎯**
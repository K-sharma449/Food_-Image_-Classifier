# 📁 Dataset Setup Guide

## Understanding Your Dataset Structure

Your downloaded dataset likely has one of these structures:

### Structure 1: With txt files (Most Common)
```
Food-10/
├── images/
│   ├── beef_tartare/
│   │   ├── 100001.jpg
│   │   ├── 100002.jpg
│   │   └── ...
│   ├── cannoli/
│   ├── ceviche/
│   └── ... (other classes)
├── train.txt          # Contains: beef_tartare/100001
├── test.txt           # Contains: beef_tartare/100050
└── ...
```

**✅ The script will automatically:**
1. Read `train.txt` and `test.txt`
2. Detect all classes (e.g., beef_tartare, cannoli, etc.)
3. Copy images from `images/class/` to `train/class/` and `test/class/`
4. Create organized structure for training

### Structure 2: Already Organized
```
Food-10/
├── train/
│   ├── beef_tartare/
│   │   ├── 100001.jpg
│   │   └── ...
│   ├── cannoli/
│   └── ...
└── test/
    ├── beef_tartare/
    └── ...
```

**✅ The script will:**
1. Detect this structure automatically
2. Use existing organization
3. Proceed directly to training

---

## Setup Instructions

### Step 1: Download Dataset

1. Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/anamikachhabra/food-items-classification-dataset-10-classes)
2. Click "Download" (requires Kaggle account)
3. Extract the ZIP file

### Step 2: Place in Project

```bash
# Extract to your project folder
cd food10-classifier/
unzip ~/Downloads/food-items-classification.zip -d Food-10/

# Verify structure
ls Food-10/
# Should show: images/ train.txt test.txt (or train/ test/)
```

### Step 3: Verify Setup

```bash
# Check dataset structure
python -c "from pathlib import Path; print(list(Path('Food-10').iterdir()))"

# Expected output should show either:
# [images, train.txt, test.txt] OR [train, test]
```

### Step 4: Run Training

```bash
# The script will automatically detect and organize
python main.py train
```

---

## What the Script Does

### Automatic Detection Process

```
1. Checks if Food-10/train and Food-10/test exist
   ├─ YES → Use existing structure
   └─ NO → Look for train.txt and images/

2. If txt files found:
   ├─ Read train.txt to get class names (e.g., "beef_tartare/100001")
   ├─ Extract unique classes dynamically
   ├─ Create train/ and test/ folders
   └─ Copy images from images/class/ to train/class/ and test/class/

3. Verify structure:
   ├─ Check all class folders have images
   ├─ Remove empty folders
   └─ Report statistics
```

### Example Output

```
📋 Organizing dataset from txt files...
🔍 Detecting classes from txt files...
✅ Found 10 classes: ['beef_tartare', 'cannoli', 'ceviche', ...]

📁 Copying training images...
✓ Copy: 100001.jpg → train/beef_tartare/
✓ Copy: 100002.jpg → train/beef_tartare/
...
✅ Train: 7500 copied, 0 missing

📁 Copying test images...
✓ Copy: 100050.jpg → test/beef_tartare/
...
✅ Test: 2500 copied, 0 missing

✅ Verified: 10 classes
📋 Classes: ['beef_tartare', 'cannoli', 'ceviche', ...]
```

---

## Common Issues & Solutions

### Issue 1: "Missing images/ folder or txt files"

**Cause**: Dataset not in expected location or format

**Solution**:
```bash
# Check what's in Food-10/
ls -la Food-10/

# If empty or wrong:
# 1. Re-extract the dataset
# 2. Ensure it's named "Food-10" exactly
# 3. Check ZIP contents before extracting
```

### Issue 2: "No class folders found in train/"

**Cause**: Already organized structure but empty folders

**Solution**:
```bash
# Remove and reorganize
rm -rf Food-10/train Food-10/test

# Run again - will use txt files
python main.py train
```

### Issue 3: "Missing: class/image_id"

**Cause**: Image files have different extensions or names

**Solution**: The script automatically tries:
- `.jpg`, `.jpeg`, `.JPG`, `.JPEG`
- `.png`, `.PNG`

If still failing, check actual image filenames:
```bash
ls Food-10/images/beef_tartare/ | head -5
```

### Issue 4: Classes detected but no images copied

**Cause**: Path or filename mismatch

**Solution**:
```bash
# Check train.txt format
head -5 Food-10/train.txt

# Expected format: class/image_id
# Example: beef_tartare/100001

# Check if images exist at expected path
ls Food-10/images/beef_tartare/
```

---

## Manual Organization (Alternative)

If automatic detection fails, you can organize manually:

```bash
# Create structure
mkdir -p Food-10/train
mkdir -p Food-10/test

# For each class, create folders
for class in beef_tartare cannoli ceviche chocolate_mousse clam_chowder crab_cakes dumplings foie_gras french_onion_soup frozen_yogurt; do
    mkdir -p Food-10/train/$class
    mkdir -p Food-10/test/$class
done

# Copy images according to train.txt and test.txt
# Read train.txt line by line and copy
while IFS= read -r line; do
    class=$(echo $line | cut -d'/' -f1)
    img=$(echo $line | cut -d'/' -f2)
    cp Food-10/images/$class/$img.jpg Food-10/train/$class/
done < Food-10/train.txt

# Repeat for test.txt
```

---

## Verifying Your Setup

### Quick Check Script

```python
# check_dataset.py
from pathlib import Path

data_dir = Path("Food-10")

print("Checking dataset structure...\n")

# Check for txt organization
if (data_dir / "train.txt").exists():
    print("✓ Found train.txt")
    with open(data_dir / "train.txt") as f:
        lines = f.readlines()
    print(f"  - {len(lines)} training entries")
    print(f"  - Sample: {lines[0].strip()}")

if (data_dir / "test.txt").exists():
    print("✓ Found test.txt")
    with open(data_dir / "test.txt") as f:
        lines = f.readlines()
    print(f"  - {len(lines)} test entries")

# Check for organized structure
train_dir = data_dir / "train"
if train_dir.exists():
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    print(f"\n✓ Found train/ with {len(classes)} classes")
    print(f"  - Classes: {classes[:5]}...")
    
test_dir = data_dir / "test"
if test_dir.exists():
    classes = [d.name for d in test_dir.iterdir() if d.is_dir()]
    print(f"✓ Found test/ with {len(classes)} classes")

# Check images folder
images_dir = data_dir / "images"
if images_dir.exists():
    classes = [d.name for d in images_dir.iterdir() if d.is_dir()]
    print(f"\n✓ Found images/ with {len(classes)} classes")
    print(f"  - Classes: {classes}")
```

Run it:
```bash
python check_dataset.py
```

---

## Expected Final Structure

After running `python main.py train`, you should have:

```
Food-10/
├── images/              # Original (can keep or delete)
├── train/              # ✓ Created by script
│   ├── beef_tartare/
│   │   ├── 100001.jpg
│   │   ├── 100002.jpg
│   │   └── ... (~750 images)
│   ├── cannoli/
│   ├── ceviche/
│   └── ... (10 classes)
│
├── test/               # ✓ Created by script
│   ├── beef_tartare/
│   │   ├── 100050.jpg
│   │   └── ... (~250 images)
│   ├── cannoli/
│   └── ... (10 classes)
│
├── train.txt          # Original split info
└── test.txt           # Original split info
```

---

## File Size Check

```bash
# Check total dataset size
du -sh Food-10/

# Check per folder
du -sh Food-10/train
du -sh Food-10/test
du -sh Food-10/images

# Count images
find Food-10/train -name "*.jpg" | wc -l
find Food-10/test -name "*.jpg" | wc -l
```

**Expected**:
- Train: ~7,500 images (750 per class × 10)
- Test: ~2,500 images (250 per class × 10)
- Total: ~10,000 images
- Size: ~2-3 GB

---

## Still Having Issues?

### Enable Debug Mode

```python
# In main.py, add at the top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Inspection

```bash
# Check train.txt format
head -10 Food-10/train.txt

# Check actual image filenames
ls Food-10/images/beef_tartare/ | head -10

# Check for hidden files
ls -la Food-10/images/beef_tartare/ | head -10
```

### Contact & Support

If issues persist:
1. Create GitHub issue with:
   - Output of `ls -R Food-10/` (first 50 lines)
   - Content of first 5 lines of train.txt
   - Error message
   - Operating system

2. Check Kaggle dataset comments for similar issues

---

## Success Indicators

You'll know setup worked when you see:

```
✅ Verified: 10 classes
📋 Classes: ['beef_tartare', 'cannoli', ...]
✅ Train: 7500 copied, 0 missing
✅ Test: 2500 copied, 0 missing

📊 Performing EDA...
✅ Saved: class_dist.png
✅ Saved: samples.png

🚀 Starting Training on cuda
```

**Happy Training! 🚀**
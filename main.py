# main.py - Fully dynamic class detection from actual dataset
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import random
from pathlib import Path
import shutil
import json

# Configuration
DATA_DIR = "Food-10"
MODEL_PATH = "model.pth"
CLASSES_PATH = "classes.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
TRAIN_T = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

VAL_T = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model(num_classes):
    """Initialize EfficientNet-B0 model"""
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    return model.to(DEVICE)

def verify_structure(data_dir):
    """Verify dataset structure and return classes"""
    train_path = Path(data_dir) / 'train'
    test_path = Path(data_dir) / 'test'
    
    if not (train_path.exists() and test_path.exists()):
        print(f"❌ Missing train/test folders in {data_dir}")
        return None
    
    train_classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    test_classes = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
    
    if not train_classes:
        print("❌ No class folders found in train/")
        return None
    
    if train_classes != test_classes:
        print(f"⚠️ Train/test class mismatch!\nTrain: {train_classes}\nTest: {test_classes}")
        # Use train classes as primary
    
    # Check for empty folders
    empty_classes = []
    for cls in train_classes:
        train_imgs = list((train_path / cls).glob('*.[jJ][pP][gG]')) + \
                     list((train_path / cls).glob('*.[jJ][pP][eE][gG]')) + \
                     list((train_path / cls).glob('*.[pP][nN][gG]'))
        if not train_imgs:
            empty_classes.append(cls)
    
    if empty_classes:
        print(f"⚠️ Empty class folders (will be skipped): {empty_classes}")
        train_classes = [c for c in train_classes if c not in empty_classes]
    
    print(f"✅ Verified: {len(train_classes)} classes")
    print(f"📋 Classes: {train_classes}")
    return train_classes

def organize_from_txt(data_dir):
    """Organize dataset from train.txt and test.txt files"""
    dataset_p = Path(data_dir)
    images_src = dataset_p / "images"
    train_txt = dataset_p / "train.txt"
    test_txt = dataset_p / "test.txt"
    
    if not all([images_src.exists(), train_txt.exists(), test_txt.exists()]):
        print("⚠️ Missing images/ folder or txt files")
        return None
    
    print("📋 Organizing dataset from txt files...")
    
    # Create train/test directories
    train_dir = dataset_p / "train"
    test_dir = dataset_p / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Extract ACTUAL classes from txt files (dynamic detection)
    print("🔍 Detecting classes from txt files...")
    all_classes = set()
    for txt_file in [train_txt, test_txt]:
        with open(txt_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    # Handle different formats: "class/id" or "class id" or just filenames
                    parts = line.replace('\\', '/').split('/')
                    if len(parts) >= 2:
                        all_classes.add(parts[0])
                    elif len(parts) == 1:
                        # Try to extract class from filename pattern
                        filename = parts[0]
                        # Common patterns: classname_id.jpg or id_classname.jpg
                        if '_' in filename:
                            # Assume format: class_id.jpg
                            class_part = filename.split('_')[0]
                            all_classes.add(class_part)
    
    classes = sorted(list(all_classes))
    num_classes = len(classes)
    print(f"✅ Found {num_classes} classes: {classes}")
    
    # Copy train images
    print("📁 Copying training images...")
    with open(train_txt) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if lines:
        print(f"📝 Sample train line: '{lines[0]}'")
    
    train_copied = 0
    train_missing = 0
    for line in tqdm(lines, desc="Train"):
        parts = line.replace('\\', '/').split('/')
        if len(parts) < 2:
            continue
        
        cls, img_id = parts[0], parts[1]
        
        # Try multiple image extensions
        found = False
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            img_name = f"{img_id}{ext}" if not img_id.endswith(ext) else img_id
            src = images_src / cls / img_name
            
            if src.exists():
                dst = train_dir / cls / img_name
                dst.parent.mkdir(exist_ok=True)
                shutil.copy2(src, dst)
                train_copied += 1
                if train_copied <= 3:
                    print(f"✓ Copy: {src.name} → train/{cls}/")
                found = True
                break
        
        if not found:
            train_missing += 1
            if train_missing <= 3:
                print(f"✗ Missing: {cls}/{img_id}")
    
    print(f"✅ Train: {train_copied} copied, {train_missing} missing")
    
    # Copy test images
    print("📁 Copying test images...")
    with open(test_txt) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if lines:
        print(f"📝 Sample test line: '{lines[0]}'")
    
    test_copied = 0
    test_missing = 0
    for line in tqdm(lines, desc="Test"):
        parts = line.replace('\\', '/').split('/')
        if len(parts) < 2:
            continue
        
        cls, img_id = parts[0], parts[1]
        
        # Try multiple image extensions
        found = False
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
            img_name = f"{img_id}{ext}" if not img_id.endswith(ext) else img_id
            src = images_src / cls / img_name
            
            if src.exists():
                dst = test_dir / cls / img_name
                dst.parent.mkdir(exist_ok=True)
                shutil.copy2(src, dst)
                test_copied += 1
                if test_copied <= 3:
                    print(f"✓ Copy: {src.name} → test/{cls}/")
                found = True
                break
        
        if not found:
            test_missing += 1
            if test_missing <= 3:
                print(f"✗ Missing: {cls}/{img_id}")
    
    print(f"✅ Test: {test_copied} copied, {test_missing} missing")
    
    return verify_structure(data_dir)

def ensure_dataset():
    """Ensure dataset is properly organized"""
    data_path = Path(DATA_DIR)
    
    # Check if already organized
    if (data_path / "train").exists() and (data_path / "test").exists():
        print("✅ Dataset structure found (train/test folders exist)")
        classes = verify_structure(DATA_DIR)
        if classes and len(classes) > 0:
            print(f"✅ Using existing organized dataset with {len(classes)} classes")
            return classes
        else:
            print("⚠️ Existing structure has issues, attempting reorganization...")
    
    # Try to organize from txt files
    if (data_path / "train.txt").exists() and (data_path / "images").exists():
        print("📋 Found train.txt and images/ folder, organizing dataset...")
        classes = organize_from_txt(DATA_DIR)
        if classes and len(classes) > 0:
            return classes
    
    raise RuntimeError(
        f"❌ Dataset setup failed!\n\n"
        f"Please ensure '{DATA_DIR}' folder contains either:\n"
        f"  Option 1: train/ and test/ subdirectories with class folders\n"
        f"  Option 2: images/ folder with train.txt and test.txt files\n\n"
        f"Current structure in '{DATA_DIR}':\n"
        f"{list(data_path.iterdir()) if data_path.exists() else 'Folder not found'}"
    )

def perform_eda(data_dir, classes):
    """Exploratory Data Analysis"""
    print("\n📊 Performing EDA...")
    train_path = Path(data_dir) / 'train'
    test_path = Path(data_dir) / 'test'
    
    # Count images per class
    train_counts = []
    test_counts = []
    for cls in classes:
        train_imgs = len(list((train_path / cls).glob('*.[jJ][pP]*'))) + \
                     len(list((train_path / cls).glob('*.[pP][nN][gG]')))
        test_imgs = len(list((test_path / cls).glob('*.[jJ][pP]*'))) + \
                    len(list((test_path / cls).glob('*.[pP][nN][gG]')))
        train_counts.append(train_imgs)
        test_counts.append(test_imgs)
    
    total_train = sum(train_counts)
    total_test = sum(test_counts)
    print(f"📊 Total training images: {total_train}")
    print(f"📊 Total test images: {total_test}")
    print(f"📊 Classes: {len(classes)}")
    
    # Plot class distribution
    plt.figure(figsize=(14, 6))
    x = np.arange(len(classes))
    width = 0.35
    plt.bar(x - width/2, train_counts, width, label='Train', alpha=0.8, color='#3b82f6')
    plt.bar(x + width/2, test_counts, width, label='Test', alpha=0.8, color='#10b981')
    plt.xlabel('Food Classes', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.title(f'Class Distribution ({len(classes)} Classes)', fontsize=14, fontweight='bold')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: class_dist.png")
    
    # Sample images
    n_samples = min(10, len(classes))
    rows = (n_samples + 4) // 5  # Dynamic rows
    cols = min(5, n_samples)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    sample_classes = random.sample(classes, n_samples)
    for i, cls in enumerate(sample_classes):
        cls_path = train_path / cls
        img_files = list(cls_path.glob('*.[jJ][pP]*')) + list(cls_path.glob('*.[pP][nN][gG]'))
        
        if img_files:
            img_path = random.choice(img_files)
            try:
                img = Image.open(img_path).convert('RGB')
                axes[i].imshow(img)
                axes[i].set_title(cls.replace('_', ' ').title(), fontsize=10)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{cls}", 
                           ha='center', va='center')
                axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: samples.png")

def train_model(epochs=5, batch_size=32, lr=1e-4):
    """Train the model"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting Training on {DEVICE}")
    print(f"{'='*60}\n")
    
    # Ensure dataset and get classes
    classes = ensure_dataset()
    num_classes = len(classes)
    
    # Save classes for later use
    with open(CLASSES_PATH, 'w') as f:
        json.dump(classes, f, indent=2)
    print(f"✅ Saved {num_classes} classes to {CLASSES_PATH}\n")
    
    # Perform EDA
    perform_eda(DATA_DIR, classes)
    
    # Load datasets
    train_path = Path(DATA_DIR) / 'train'
    test_path = Path(DATA_DIR) / 'test'
    
    train_ds = datasets.ImageFolder(str(train_path), TRAIN_T)
    test_ds = datasets.ImageFolder(str(test_path), VAL_T)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"📚 Loaded: {len(train_ds)} train, {len(test_ds)} test images")
    print(f"🏷️  Classes ({num_classes}): {', '.join(classes[:5])}{'...' if num_classes > 5 else ''}\n")
    
    # Initialize model
    model = get_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = 0.0
    history = {'train_loss': [], 'val_f1': []}
    
    # Training loop
    for epoch in range(epochs):
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'─'*60}")
        
        # Train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Training")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        scheduler.step()
        
        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        history['val_f1'].append(macro_f1)
        
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Loss: {avg_loss:.4f} | Macro F1: {macro_f1:.4f}")
        
        # Classification report
        report = classification_report(all_labels, all_preds, target_names=classes, 
                                      zero_division=0)
        print("\n" + report)
        
        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 ✨ New best model saved! F1: {best_f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(max(10, num_classes), max(8, num_classes)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - Epoch {epoch+1} (F1: {macro_f1:.3f})', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'cm_epoch{epoch+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Training history plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, epochs+1)
    ax1.plot(epochs_range, history['train_loss'], marker='o', linewidth=2, color='#ef4444')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, history['val_f1'], marker='o', linewidth=2, color='#10b981')
    ax2.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"🎉 Training Complete!")
    print(f"{'='*60}")
    print(f"📈 Best Macro F1: {best_f1:.4f}")
    print(f"💾 Model saved to: {MODEL_PATH}")
    print(f"📁 Classes saved to: {CLASSES_PATH}")
    print(f"📊 Visualizations saved: class_dist.png, samples.png, cm_epoch*.png, training_history.png")
    print(f"{'='*60}\n")

def predict_image(image_path):
    """Predict single image"""
    if not Path(MODEL_PATH).exists():
        print("❌ Model not found! Please train first: python main.py train")
        return None
    
    if not Path(CLASSES_PATH).exists():
        print("❌ Classes file not found! Please train first.")
        return None
    
    # Load classes
    with open(CLASSES_PATH) as f:
        classes = json.load(f)
    
    # Load model
    model = get_model(len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Process image
    try:
        img = Image.open(image_path).convert('RGB')
        x = VAL_T(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, 1)[0]
            conf, idx = torch.max(probs, 0)
        
        result = {
            'prediction': classes[idx],
            'confidence': conf.item(),
            'all_probabilities': {classes[i]: float(probs[i]) for i in range(len(classes))}
        }
        
        print(f"\n{'='*60}")
        print(f"🍽️  Prediction Results")
        print(f"{'='*60}")
        print(f"📸 Image: {Path(image_path).name}")
        print(f"🏆 Predicted: {result['prediction'].upper().replace('_', ' ')}")
        print(f"✅ Confidence: {result['confidence']:.2%}")
        print(f"\n📊 Top 5 Probabilities:")
        sorted_probs = sorted(result['all_probabilities'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        for i, (cls, prob) in enumerate(sorted_probs, 1):
            print(f"  {i}. {cls.replace('_', ' ').title():20s}: {prob:.2%}")
        print(f"{'='*60}\n")
        
        return result
    
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Food Classifier - Dynamic Class Detection')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict single image')
    predict_parser.add_argument('--image', required=True, help='Path to image file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.command == 'predict':
        predict_image(args.image)
    else:
        parser.print_help()
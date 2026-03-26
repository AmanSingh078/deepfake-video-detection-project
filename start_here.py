"""
Tri-Expert Detection Suite - Quick Start Guide
========================================

This project detects fake/deepfake videos using pre-trained models.

SETUP STEPS:
============

1. INSTALL DEPENDENCIES (Run this first):
   py -m pip install torch torchvision albumentations timm facenet-pytorch pandas numpy opencv-python pillow tqdm

2. DOWNLOAD PRE-TRAINED WEIGHTS:
   Option A - Manual download (Windows):
   - Obtain model weights for final analysis
   - Place files in: weights/ folder
   
   Files needed:
   - final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36
   - final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19
   - final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29
   - final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31
   - final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37
   - final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40
   - final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23

3. RUN DETECTION:
   Create a folder with test videos (.mp4 files) then run:
   
   py predict_folder.py --test-dir "path\to\videos" --output results.csv
   
   Example:
   py predict_folder.py --test-dir "C:\Videos\Test" --output submission.csv

4. VIEW RESULTS:
   Open results.csv to see predictions (0=fake, 1=real)


FOR TRAINING (Advanced):
=========================
1. Preprocess data: py preprocessing/detect_original_faces.py --root-dir <data_root>
2. Train model: See train.sh script for multi-GPU training setup

"""

import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    print("=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'cv2': 'OpenCV',
        'pandas': 'Pandas',
        'albumentations': 'Albumentations',
        'timm': 'Timm',
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            module = sys.modules[package]
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {name:20s} v{version}")
        except ImportError:
            print(f"✗ {name:20s} NOT INSTALLED")
            missing.append(name)
    
    print("=" * 70)
    
    if missing:
        print("\n❌ MISSING PACKAGES:")
        print(f"   Run: py -m pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ ALL PACKAGES INSTALLED!")
        return True

def check_weights():
    """Check if model weights are downloaded"""
    print("\n" + "=" * 70)
    print("CHECKING MODEL WEIGHTS")
    print("=" * 70)
    
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        print(f"✗ Weights directory '{weights_dir}' not found")
        return False
    
    weight_files = [
        "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
        "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
        "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
        "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
        "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
        "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
        "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
    ]
    
    found = 0
    for wf in weight_files:
        full_path = os.path.join(weights_dir, wf)
        if os.path.exists(full_path):
            print(f"✓ {wf}")
            found += 1
        else:
            print(f"✗ {wf} - MISSING")
    
    print("=" * 70)
    print(f"Weights found: {found}/{len(weight_files)}")
    
    if found == len(weight_files):
        print("\n✅ ALL WEIGHTS DOWNLOADED!")
        return True
    else:
        print("\n❌ SOME WEIGHTS MISSING")
        print("   Download and place missing model binary weights in the 'weights' folder")
        return False

def main():
    print("\n🛡️ TRI-EXPERT DETECTION PROJECT - STARTUP CHECK\n")
    
    deps_ok = check_dependencies()
    weights_ok = check_weights()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    if not deps_ok:
        print("\n1️⃣ Install dependencies first:")
        print("   py -m pip install torch torchvision albumentations timm facenet-pytorch pandas numpy opencv-python pillow tqdm")
    
    if deps_ok and not weights_ok:
        print("\n2️⃣ Download model weights:")
        print("   Download model binary files and place them in the 'weights' folder")
        print("   Download all 7 weight files and place them in the 'weights' folder")
    
    if deps_ok and weights_ok:
        print("\n✅ READY TO RUN!")
        print("\nTo detect fake videos:")
        print("   py predict_folder.py --test-dir \"path\\to\\your\\videos\" --output results.csv")
        print("\nExample:")
        print("   py predict_folder.py --test-dir \"C:\\Videos\\Test\" --output submission.csv")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

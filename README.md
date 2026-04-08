# TriExpert Deepfake Detection System

A professional deepfake detection system using ensemble learning with EfficientNet-B7 to identify AI-generated video content with state-of-the-art accuracy.

---

## System Overview

The TriExpert Detection Suite uses three independent EfficientNet-B7 models trained with different random seeds (111, 555, 777) to achieve robust deepfake detection through ensemble voting.

### Architecture

- **Backbone**: EfficientNet-B7 (tf_efficientnet_b7_ns)
- **Input**: 380x380 RGB images
- **Ensemble**: 3 models with weighted voting [0.36, 0.32, 0.32]
- **Face Detection**: MTCNN with temporal smoothing
- **Performance**: 99.12% accuracy on DFDC dataset

---

## Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended)
- PyTorch 1.10+

### Installation

1. Install dependencies:
```bash
pip install torch torchvision opencv-python albumentations timm facenet-pytorch tqdm pandas scikit-learn flask
```

2. Download model weights:
```bash
python download_weights.py
```

### Usage

**Single video prediction:**
```bash
python demo.py --video test.mp4 --weights \
    weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36 \
    weights/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19 \
    weights/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29
```

**Batch processing:**
```bash
python predict_folder.py --test-dir /path/to/videos --output results.csv
```

**Web interface:**
```bash
python server.py
```
Then open `http://localhost:5000` in your browser.

---

## Training

To train the TriExpert models from scratch:

```bash
# Train all 3 models (requires GPU)
bash train.sh /path/to/dataset 1
```

---

## Project Structure

```
├── configs/                 # Configuration files
├── preprocessing/           # Face detection and video processing
├── training/               # Training pipelines and models
│   ├── datasets/           # Data loading utilities
│   ├── pipelines/          # Training scripts
│   ├── sgd/                # Optimizer utilities
│   └── zoo/                # Model architectures
├── weights/                # Model checkpoints (download separately)
├── demo.py                 # Single video prediction
├── predict_folder.py       # Batch prediction
├── server.py               # Web interface
├── download_weights.py     # Weight downloader
└── train.sh                # Training script
```

---

## Core Technologies

- **Deep Learning**: PyTorch
- **Architecture**: EfficientNet-B7 (timm)
- **Data Augmentation**: Albumentations
- **Face Detection**: MTCNN (facenet-pytorch)
- **Web Framework**: Flask

---

## License

See LICENSE file for details.

---

Developed for professional AI content verification and forensic analysis.

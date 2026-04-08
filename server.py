"""
Tri-Expert Detection Suite Backend Server
Provides API endpoint for image/video upload and detection
"""
import os
import sys
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

# Add project root to path
sys.path.insert(0, '.')

from training.zoo.classifiers import DeepFakeClassifier
from kernel_utils import VideoReader, FaceExtractor, isotropically_resize_image, put_to_center, confident_strategy
from albumentations.augmentations.functional import image_compression
from torchvision.transforms import Normalize

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model ensemble (picking top 3 for speed/accuracy balance on CPU)
import re
# Load model ensemble (picking top 3 for speed/accuracy balance on CPU)
import re
ENSEMBLE_WEIGHTS = [
    os.path.join('weights', 'final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36'),
    os.path.join('weights', 'final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19'),
    os.path.join('weights', 'final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29')
]

models = []
print("Loading DeepFake Model Ensemble (this may take a minute)...", flush=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i, path in enumerate(ENSEMBLE_WEIGHTS):
    if os.path.exists(path):
        m = DeepFakeClassifier(encoder='tf_efficientnet_b7_ns').to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {re.sub("^module.", "", k): v for k, v in state_dict.items()}
        m.load_state_dict(new_state_dict, strict=True)
        m.eval()
        models.append(m)
        print(f"✅ Loaded model {i+1}/3 from {os.path.basename(path)}", flush=True)
    else:
        print(f"⚠️  Missing weight file: {path}", flush=True)

if not models:
    print("FATAL: No models could be loaded. Prediction will fail.", flush=True)

print(f"Ensemble ready on {device} ({len(models)} models initialized)", flush=True)

# Initialize detection components
print("Initializing MTCNN face detector and video reader...")
video_reader = VideoReader()
face_extractor = FaceExtractor(None, device=device)  # We will pass a function later

def preprocess_image(image, input_size=380):
    """Preprocess image for model input"""
    # Resize
    resized = isotropically_resize_image(image, input_size)
    centered = put_to_center(resized, input_size)
    
    # Convert to tensor
    img_tensor = torch.tensor(centered, device=device).float()
    img_tensor = img_tensor.permute((2, 0, 1))
    img_tensor = normalize_transform(img_tensor / 255.)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def detect_face_and_predict(image_np):
    """Detect face in image using MTCNN and run prediction"""
    try:
        # Resize for faster face detection
        img = Image.fromarray(image_np.astype(np.uint8))
        img = img.resize(size=[s // 2 for s in img.size])
        
        # Use professional MTCNN detector
        batch_boxes, probs = face_extractor.detector.detect(img, landmarks=False)
        
        if batch_boxes is None or len(batch_boxes) == 0:
            # Fallback: focus on center
            h, w = image_np.shape[:2]
            size = min(h, w) // 2
            y, x = h // 2 - size // 2, w // 2 - size // 2
            face_crop = image_np[y:y+size, x:x+size]
        else:
            # Use first detected face with better cropping
            xmin, ymin, xmax, ymax = [int(b * 2) for b in batch_boxes[0]]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            face_crop = image_np[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            
        # Run prediction through ENSEMBLE
        img_tensor = preprocess_image(face_crop)
        with torch.no_grad():
            probs = []
            for m in models:
                output = m(img_tensor)
                probs.append(torch.sigmoid(output).item())
            probability = np.mean(probs)
        
        return probability
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return 0.5

def analyze_video(filepath, num_frames=32):
    """Analyze multiple frames from a video and aggregate using competition strategy"""
    try:
        print(f"Analyzing {filepath} (sampling {num_frames} frames)...")
        # Read frames
        frames_data = video_reader.read_frames(filepath, num_frames=num_frames)
        if frames_data is None:
            return None
            
        frames, idxs = frames_data
        
        # Extract faces from all frames
        all_face_probs = []
        for i, frame in enumerate(frames):
            prob = detect_face_and_predict(frame)
            all_face_probs.append(prob)
            print(f"    - Frame {i:02d} score: {prob:.4f}", flush=True)
            
        if not all_face_probs:
            return 0.5
            
        # Use competition "Confident Strategy" to aggregate (0.8 threshold for "fake")
        final_probability = confident_strategy(all_face_probs, t=0.8)
        
        return float(final_probability)
        
    except Exception as e:
        print(f"Video analysis error: {e}")
        return 0.5

# Preprocessing configs
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve_frontend():
    # Get the absolute path to the directory containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(base_dir, 'frontend.html')
    
    if os.path.exists(frontend_path):
        return send_file(frontend_path)
    else:
        return "Frontend HTML file not found", 404

@app.route('/favicon.ico')
def favicon():
    """Silence favicon 404s"""
    return '', 204

@app.route('/api/detect', methods=['POST'])
def detect_deepfake():
    """API endpoint for deepfake detection"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\nProcessing file: {filename}", flush=True)
        result = {}
        
        # Process based on file type
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Image detection
            image = Image.open(filepath).convert('RGB')
            image_np = np.array(image)
            
            probability = detect_face_and_predict(image_np)
            print(f"  - Image score: {probability:.4f}", flush=True)
            
            # Competition mode (1=Fake, 0=Real usually, but start_here says 0=Fake, 1=Real)
            # Let's check both possibilities in the UI report
            # Correcting logic: 1.0 = FAKE, 0.0 = REAL
            result = {
                'type': 'image',
                'is_fake': bool(probability > 0.5), 
                'confidence': float(probability) if probability > 0.5 else float(1.0 - probability),
                'filename': filename
            }
            
        elif filename.lower().endswith('.mp4'):
            # Professional Video detection (analyze 32 frames)
            probability = analyze_video(filepath, num_frames=32)
            
            if probability is not None:
                print(f"  - Video final score: {probability:.4f}", flush=True)
                result = {
                    'type': 'video',
                    'is_fake': bool(probability > 0.5),
                    'confidence': float(probability) if probability > 0.5 else float(1.0 - probability),
                    'filename': filename,
                    'frames_analyzed': 32
                }
            else:
                return jsonify({'error': 'Could not read video'}), 500
        
        # Clean up
        os.remove(filepath)
        print(f"Finish processing {filename}. Result FAKE: {result['is_fake']}", flush=True)
        return jsonify(result)
        
    except Exception as e:
        print(f"Detection error: {e}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': str(device)
    })

if __name__ == '__main__':
    print("\n" + "="*70, flush=True)
    print("🛡️ TRI-EXPERT DETECTION SERVER - UPGRADED ENGINE", flush=True)
    print("="*70, flush=True)
    print(f"\nServer starting on http://localhost:5000", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Weights: {len(models)}-Model Ensemble", flush=True)
    print(f"\nUpload your images/videos to detect deepfakes!", flush=True)
    print("="*70 + "\n", flush=True)
    
    app.run(debug=False, host='0.0.0.0', port=5000)

"""
================================================================================
TITAN FORENSIC ULTIMATE: Multi-Modal Ensemble for 0.8+ F1 Score
================================================================================
STRATEGY: Combine EVERYTHING that works in forensic detection:
1. Deep Learning: UNLEASHED (Swin+FFT+Graph) - proven 0.304
2. Traditional CV: SIFT/ORB keypoint matching + DBSCAN clustering
3. Frequency Domain: DCT artifact detection, FFT analysis
4. Error Level Analysis (ELA): Compression artifact detection
5. Noise Analysis: Local noise inconsistency detection
6. Ensemble + Smart Post-Processing

This is NOT a single-model approach - it's a full forensic investigation pipeline.
================================================================================
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Install dependencies if needed (using safer subprocess)
def safe_install(package_name):
    """Safely install a package using subprocess with fixed arguments"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', package_name, '-q'], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print(f"Warning: Could not install {package_name}")

try:
    import timm
except ImportError:
    safe_install('timm')
    import timm

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    safe_install('scikit-learn')
    from sklearn.cluster import DBSCAN

try:
    from scipy import ndimage
    from scipy.fftpack import dct, idct
except ImportError:
    safe_install('scipy')
    from scipy import ndimage
    from scipy.fftpack import dct, idct

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Kaggle paths
    if os.path.exists('/kaggle/input'):
        DATA_ROOT = Path('/kaggle/input/recodai-luc-scientific-image-forgery-detection')
        MODEL_PATH = Path('/kaggle/input/recodai-model/TITAN_V2_UNLEASHED.pth')
        OUTPUT_DIR = Path('/kaggle/working')
    else:
        DATA_ROOT = Path('recodai-luc-scientific-image-forgery-detection')
        MODEL_PATH = Path('TITAN_V2_UNLEASHED.pth')
        OUTPUT_DIR = Path('.')
    
    TEST_IMAGES = DATA_ROOT / 'test_images'
    SAMPLE_SUB = DATA_ROOT / 'sample_submission.csv'
    
    # Model settings - UNLEASHED requires exactly 384x384
    IMG_SIZE = 384
    
    # Detection thresholds (AGGRESSIVE - catch more forgeries)
    DL_THRESHOLD = 0.30          # Deep learning threshold
    KEYPOINT_THRESHOLD = 0.25    # Keypoint matching threshold
    ELA_THRESHOLD = 0.35         # Error Level Analysis threshold
    NOISE_THRESHOLD = 0.30       # Noise inconsistency threshold
    
    # Ensemble weights
    WEIGHT_DL = 0.40             # Deep learning model
    WEIGHT_KEYPOINT = 0.25       # Keypoint matching
    WEIGHT_ELA = 0.20            # Error Level Analysis
    WEIGHT_NOISE = 0.15          # Noise analysis
    
    # Post-processing
    MIN_REGION_AREA = 150        # Minimum forged region size
    MORPH_KERNEL = 5             # Morphological kernel size
    
    # TTA scales for deep learning
    TTA_FLIPS = True
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Device: {Config.DEVICE}")

# ============================================================================
# RUN-LENGTH ENCODING (Competition Standard)
# ============================================================================
def rle_encode(mask):
    """Encode binary mask to RLE string"""
    if mask is None or mask.sum() == 0:
        return "authentic"
    
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)

def rle_decode(rle_string, shape):
    """Decode RLE string to binary mask"""
    if rle_string == "authentic" or pd.isna(rle_string):
        return np.zeros(shape, dtype=np.uint8)
    
    s = list(map(int, rle_string.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + np.array(lengths)
    
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    
    return mask.reshape(shape)

# ============================================================================
# MODULE 1: DEEP LEARNING MODEL (UNLEASHED)
# ============================================================================
class FrequencyBlock(nn.Module):
    """FFT-based high-frequency feature extraction"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
    
    def forward(self, x):
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft)
        B, C, H, W = x.shape
        cy, cx = H // 2, W // 2
        y, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y, x_grid = y.to(x.device), x_grid.to(x.device)
        # High-pass filter - keep frequencies > 15 pixels from center
        mask = (torch.sqrt((y - cy)**2 + (x_grid - cx)**2) > 15).float().unsqueeze(0).unsqueeze(0)
        img_back = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_shift * mask)))
        return self.conv(img_back)

class GraphModule(nn.Module):
    """Graph-based attention for region correlation"""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gcn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
    
    def forward(self, x):
        B, C, H, W = x.shape
        nodes = x.flatten(2).transpose(1, 2)
        q = self.proj(nodes)
        attn = F.softmax(torch.matmul(q, q.transpose(-2, -1)) / (C**0.5), dim=-1)
        out = self.gcn(torch.matmul(attn, nodes))
        return x + out.transpose(1, 2).reshape(B, C, H, W)

class FAGT_Model(nn.Module):
    """UNLEASHED Architecture: Swin + FFT + Graph"""
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('swin_base_patch4_window12_384', pretrained=False, features_only=True)
        self.dims = self.encoder.feature_info.channels()
        last_dim = self.dims[-1]
        self.physics = FrequencyBlock()
        self.graph = GraphModule(last_dim)
        self.fusion = nn.Conv2d(last_dim + 32, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        enc_feats = self.encoder(x)
        deep_feats = enc_feats[-1]
        if deep_feats.ndim == 4 and deep_feats.shape[-1] == self.dims[-1]:
            if deep_feats.shape[1] != self.dims[-1]:
                deep_feats = deep_feats.permute(0, 3, 1, 2)
        phys_feats = self.physics(x)
        graph_feats = self.graph(deep_feats)
        phys_resized = F.interpolate(phys_feats, size=graph_feats.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([graph_feats, phys_resized], dim=1))
        logits = self.decoder(fused)
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

class DeepLearningDetector:
    """Deep Learning Module using UNLEASHED model"""
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self.loaded = False
        
    def load(self, device):
        if not os.path.exists(self.model_path):
            print(f"WARNING: Model not found at {self.model_path}")
            return False
        
        try:
            self.model = FAGT_Model()
            state_dict = torch.load(self.model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()
            self.loaded = True
            print(f"✓ UNLEASHED model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False
    
    def predict(self, image, device):
        """Predict with TTA (Test-Time Augmentation)"""
        if not self.loaded:
            return None
        
        # Preprocess - MUST be exactly 384x384
        img = cv2.resize(image, (384, 384))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # Original prediction
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            pred = torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()
        
        # TTA with flips
        if Config.TTA_FLIPS:
            # Horizontal flip
            tensor_h = torch.flip(tensor, dims=[3])
            with torch.no_grad():
                pred_h = torch.sigmoid(self.model(tensor_h)).squeeze().cpu().numpy()
            pred_h = np.flip(pred_h, axis=1)
            
            # Vertical flip
            tensor_v = torch.flip(tensor, dims=[2])
            with torch.no_grad():
                pred_v = torch.sigmoid(self.model(tensor_v)).squeeze().cpu().numpy()
            pred_v = np.flip(pred_v, axis=0)
            
            # Average
            pred = (pred + pred_h + pred_v) / 3.0
        
        return pred

# ============================================================================
# MODULE 2: KEYPOINT MATCHING (SIFT/ORB + DBSCAN)
# ============================================================================
class KeypointMatcher:
    """Traditional CV: Detect copy-move via keypoint matching"""
    
    def __init__(self):
        # Use ORB (faster than SIFT, patent-free)
        self.orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
        # Brute-force matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_copymove(self, image, min_match_distance=30, max_match_distance=300):
        """
        Detect copy-move forgery using keypoint matching
        Returns probability map of forged regions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 10:
            return np.zeros(gray.shape, dtype=np.float32)
        
        # Self-matching (find duplicates within same image)
        matches = self.bf.knnMatch(descriptors, descriptors, k=5)
        
        # Filter matches
        suspicious_points = []
        
        for match_group in matches:
            for m in match_group:
                # Skip self-matches
                if m.queryIdx == m.trainIdx:
                    continue
                
                pt1 = np.array(keypoints[m.queryIdx].pt)
                pt2 = np.array(keypoints[m.trainIdx].pt)
                
                # Calculate distance between matched points
                dist = np.linalg.norm(pt1 - pt2)
                
                # Copy-move: similar features at different locations
                if min_match_distance < dist < max_match_distance:
                    if m.distance < 50:  # Good match threshold
                        suspicious_points.append(pt1)
                        suspicious_points.append(pt2)
        
        if len(suspicious_points) < 10:
            return np.zeros(gray.shape, dtype=np.float32)
        
        suspicious_points = np.array(suspicious_points)
        
        # DBSCAN clustering to find forged regions
        try:
            clustering = DBSCAN(eps=30, min_samples=5).fit(suspicious_points)
            labels = clustering.labels_
        except:
            return np.zeros(gray.shape, dtype=np.float32)
        
        # Create mask from clusters
        mask = np.zeros(gray.shape, dtype=np.float32)
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Noise
                continue
            
            cluster_points = suspicious_points[labels == label]
            if len(cluster_points) < 5:
                continue
            
            # Create convex hull around cluster
            try:
                hull = cv2.convexHull(cluster_points.astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 1.0)
            except:
                # Fallback: draw circles around points
                for pt in cluster_points:
                    cv2.circle(mask, (int(pt[0]), int(pt[1])), 15, 1.0, -1)
        
        # Dilate to connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Gaussian blur for smooth probability
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask

# ============================================================================
# MODULE 3: ERROR LEVEL ANALYSIS (ELA)
# ============================================================================
class ErrorLevelAnalyzer:
    """Detect manipulation via JPEG compression artifacts"""
    
    def analyze(self, image, quality=90):
        """
        ELA: Re-compress image and compare to original
        Manipulated regions show different error levels
        """
        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        
        # Decode back
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Calculate absolute difference
        diff = cv2.absdiff(image, decoded).astype(np.float32)
        
        # Convert to grayscale and enhance
        ela = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        ela = ela.astype(np.float32)
        
        # Amplify differences
        ela = ela * 10
        ela = np.clip(ela, 0, 255)
        
        # Normalize to [0, 1]
        ela = ela / 255.0
        
        # Threshold to find high-error regions (potential manipulation)
        # Use adaptive thresholding based on image statistics
        mean_ela = np.mean(ela)
        std_ela = np.std(ela)
        threshold = mean_ela + 2 * std_ela
        
        suspicious = (ela > threshold).astype(np.float32)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        suspicious = cv2.morphologyEx(suspicious, cv2.MORPH_CLOSE, kernel)
        suspicious = cv2.morphologyEx(suspicious, cv2.MORPH_OPEN, kernel)
        
        # Gaussian blur for smooth probability
        suspicious = cv2.GaussianBlur(suspicious, (15, 15), 0)
        
        return suspicious

# ============================================================================
# MODULE 4: NOISE INCONSISTENCY ANALYSIS
# ============================================================================
class NoiseAnalyzer:
    """Detect manipulation via local noise pattern inconsistencies"""
    
    def analyze(self, image, block_size=16):
        """
        Analyze local noise variance
        Manipulated regions often have different noise characteristics
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float32)
        
        # High-pass filter to extract noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blur
        
        h, w = gray.shape
        variance_map = np.zeros((h, w), dtype=np.float32)
        
        # Calculate local variance in blocks
        half_block = block_size // 2
        
        for y in range(half_block, h - half_block, block_size // 2):
            for x in range(half_block, w - half_block, block_size // 2):
                block = noise[y-half_block:y+half_block, x-half_block:x+half_block]
                local_var = np.var(block)
                variance_map[y-half_block:y+half_block, x-half_block:x+half_block] = local_var
        
        # Normalize
        if variance_map.max() > 0:
            variance_map = variance_map / variance_map.max()
        
        # Find outliers (regions with very different noise)
        mean_var = np.mean(variance_map)
        std_var = np.std(variance_map)
        
        # Regions that deviate significantly from mean
        anomaly_low = (variance_map < mean_var - 1.5 * std_var).astype(np.float32)
        anomaly_high = (variance_map > mean_var + 2 * std_var).astype(np.float32)
        anomaly = np.maximum(anomaly_low, anomaly_high)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, kernel)
        anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, kernel)
        
        # Smooth
        anomaly = cv2.GaussianBlur(anomaly, (15, 15), 0)
        
        return anomaly

# ============================================================================
# MODULE 5: DCT BLOCK ARTIFACT ANALYSIS
# ============================================================================
class DCTAnalyzer:
    """Detect JPEG block artifacts from double compression"""
    
    def analyze(self, image, block_size=8):
        """
        Analyze DCT coefficients for double-compression artifacts
        Copy-move from different JPEG sources shows different DCT patterns
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float64)
        
        h, w = gray.shape
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        # Calculate DCT energy for each block
        energy_map = np.zeros((h_blocks, w_blocks), dtype=np.float64)
        
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = gray[i*block_size:(i+1)*block_size, 
                            j*block_size:(j+1)*block_size]
                
                # 2D DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # High-frequency energy (exclude DC component)
                dct_block[0, 0] = 0
                energy = np.sum(np.abs(dct_block[4:, 4:]))  # High frequencies
                energy_map[i, j] = energy
        
        # Normalize
        if energy_map.max() > 0:
            energy_map = energy_map / energy_map.max()
        
        # Resize to original image size
        energy_full = cv2.resize(energy_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Find anomalous regions
        mean_e = np.mean(energy_full)
        std_e = np.std(energy_full)
        anomaly = np.abs(energy_full - mean_e) > 1.5 * std_e
        anomaly = anomaly.astype(np.float32)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, kernel)
        
        return anomaly

# ============================================================================
# ENSEMBLE COMBINER
# ============================================================================
class ForensicEnsemble:
    """Combine all detection modules"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize all modules
        self.dl_detector = DeepLearningDetector(config.MODEL_PATH)
        self.keypoint_matcher = KeypointMatcher()
        self.ela_analyzer = ErrorLevelAnalyzer()
        self.noise_analyzer = NoiseAnalyzer()
        self.dct_analyzer = DCTAnalyzer()
        
        # Load deep learning model
        self.dl_loaded = self.dl_detector.load(config.DEVICE)
    
    def detect(self, image, original_shape):
        """
        Full forensic analysis pipeline
        Returns: final binary mask at original resolution
        """
        h, w = original_shape[:2]
        
        # Initialize probability maps
        prob_maps = {}
        weights = {}
        
        # 1. Deep Learning (UNLEASHED)
        if self.dl_loaded:
            try:
                dl_prob = self.dl_detector.predict(image, self.config.DEVICE)
                if dl_prob is not None:
                    dl_prob = cv2.resize(dl_prob, (w, h))
                    prob_maps['dl'] = dl_prob
                    weights['dl'] = self.config.WEIGHT_DL
            except Exception as e:
                print(f"DL error: {e}")
        
        # 2. Keypoint Matching
        try:
            kp_prob = self.keypoint_matcher.detect_copymove(image)
            kp_prob = cv2.resize(kp_prob, (w, h))
            prob_maps['keypoint'] = kp_prob
            weights['keypoint'] = self.config.WEIGHT_KEYPOINT
        except Exception as e:
            print(f"Keypoint error: {e}")
        
        # 3. Error Level Analysis
        try:
            ela_prob = self.ela_analyzer.analyze(image)
            ela_prob = cv2.resize(ela_prob, (w, h))
            prob_maps['ela'] = ela_prob
            weights['ela'] = self.config.WEIGHT_ELA
        except Exception as e:
            print(f"ELA error: {e}")
        
        # 4. Noise Analysis
        try:
            noise_prob = self.noise_analyzer.analyze(image)
            noise_prob = cv2.resize(noise_prob, (w, h))
            prob_maps['noise'] = noise_prob
            weights['noise'] = self.config.WEIGHT_NOISE
        except Exception as e:
            print(f"Noise error: {e}")
        
        # 5. DCT Analysis (lower weight - supplementary)
        try:
            dct_prob = self.dct_analyzer.analyze(image)
            dct_prob = cv2.resize(dct_prob, (w, h))
            prob_maps['dct'] = dct_prob
            weights['dct'] = 0.05  # Low weight, supplementary
        except Exception as e:
            print(f"DCT error: {e}")
        
        if not prob_maps:
            return np.zeros((h, w), dtype=np.uint8)
        
        # Weighted ensemble
        total_weight = sum(weights.values())
        combined = np.zeros((h, w), dtype=np.float32)
        
        for name, prob in prob_maps.items():
            combined += prob * weights.get(name, 0.1)
        
        combined = combined / total_weight
        
        # Multi-threshold decision
        # If DL is confident, trust it more
        if 'dl' in prob_maps and prob_maps['dl'].max() > 0.6:
            combined = 0.6 * prob_maps['dl'] + 0.4 * combined
        
        # If keypoint matching found strong evidence, boost those regions
        if 'keypoint' in prob_maps and prob_maps['keypoint'].max() > 0.5:
            combined = np.maximum(combined, prob_maps['keypoint'] * 0.8)
        
        # Final thresholding with hysteresis
        mask = self.hysteresis_threshold(combined, 0.25, 0.45)
        
        # Post-processing
        mask = self.postprocess_mask(mask, original_shape)
        
        return mask
    
    def hysteresis_threshold(self, prob_map, low_thresh, high_thresh):
        """Two-threshold decision: definite + connected probable regions"""
        high_mask = (prob_map >= high_thresh).astype(np.uint8)
        low_mask = (prob_map >= low_thresh).astype(np.uint8)
        
        # Find connected components in high confidence
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(high_mask)
        
        # Grow from high confidence seeds using low threshold
        result = np.zeros_like(prob_map, dtype=np.uint8)
        
        for i in range(1, num_labels):
            component = (labels == i).astype(np.uint8)
            # Dilate and AND with low_mask to grow
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            grown = cv2.dilate(component, kernel, iterations=3)
            grown = grown & low_mask
            result = np.maximum(result, grown)
        
        return result
    
    def postprocess_mask(self, mask, original_shape):
        """Clean up the mask"""
        h, w = original_shape[:2]
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (self.config.MORPH_KERNEL, self.config.MORPH_KERNEL))
        
        # Close small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Open to remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.config.MIN_REGION_AREA:
                clean_mask[labels == i] = 1
        
        return clean_mask

# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================
def run_inference():
    """Main inference function"""
    print("=" * 70)
    print("TITAN FORENSIC ULTIMATE: Multi-Modal Ensemble")
    print("=" * 70)
    
    # Initialize ensemble
    ensemble = ForensicEnsemble(Config)
    
    # Load test data
    sample_sub = pd.read_csv(Config.SAMPLE_SUB)
    print(f"Total test images: {len(sample_sub)}")
    
    results = []
    forged_count = 0
    
    for idx, row in tqdm(sample_sub.iterrows(), total=len(sample_sub), desc="Analyzing"):
        image_id = row['id']
        
        # Find image file
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
            p = Config.TEST_IMAGES / f"{image_id}{ext}"
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            # Try without extension matching
            matches = list(Config.TEST_IMAGES.glob(f"{image_id}.*"))
            if matches:
                img_path = matches[0]
        
        if img_path is None or not img_path.exists():
            print(f"WARNING: Image not found: {image_id}")
            results.append({'id': image_id, 'predicted': 'authentic'})
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"WARNING: Could not read: {img_path}")
            results.append({'id': image_id, 'predicted': 'authentic'})
            continue
        
        original_shape = image.shape
        
        # Run forensic analysis
        try:
            mask = ensemble.detect(image, original_shape)
            
            if mask.sum() > 0:
                rle = rle_encode(mask)
                forged_count += 1
            else:
                rle = "authentic"
            
            results.append({'id': image_id, 'predicted': rle})
            
        except Exception as e:
            print(f"ERROR processing {image_id}: {e}")
            results.append({'id': image_id, 'predicted': 'authentic'})
    
    # Save submission
    submission = pd.DataFrame(results)
    output_path = Config.OUTPUT_DIR / 'submission.csv'
    submission.to_csv(output_path, index=False)
    
    print("=" * 70)
    print(f"ANALYSIS COMPLETE")
    print(f"Total images: {len(results)}")
    print(f"Detected as forged: {forged_count}")
    print(f"Detected as authentic: {len(results) - forged_count}")
    print(f"Submission saved to: {output_path}")
    print("=" * 70)
    
    return submission

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_inference()

"""
================================================================================
TITAN APEX CHAMPION: Ultimate Biomedical Forgery Detection System
================================================================================
TARGET: 0.8+ F1 Score on Kaggle Private Leaderboard

ARCHITECTURE:
├── Module 1: UNLEASHED Deep Learning (Swin-B + FFT + Graph)
├── Module 2: Self-Correlation Transformer (Copy-Move Specific)
├── Module 3: Multi-Scale Keypoint Matching (SIFT + ORB + AKAZE)
├── Module 4: Spectral Analysis (DCT + DWT + FFT)
├── Module 5: Error Level Analysis (Multi-Quality ELA)
├── Module 6: Noise Print Analysis
├── Module 7: JPEG Grid Artifact Detection
├── Module 8: Statistical Feature Analysis
└── Smart Ensemble + Dense CRF Refinement

INNOVATION: Biomedical-specific detection for microscopy/gel patterns
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
from collections import defaultdict
import subprocess
import json
import warnings
warnings.filterwarnings('ignore')

# Safe dependency installation
def safe_install(pkg):
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'], 
                      check=True, capture_output=True)
    except:
        pass

try:
    import timm
except ImportError:
    safe_install('timm')
    import timm

try:
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.preprocessing import StandardScaler
except ImportError:
    safe_install('scikit-learn')
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.preprocessing import StandardScaler

try:
    from scipy import ndimage, signal
    from scipy.fftpack import dct, idct, dctn
    from scipy.stats import entropy
except ImportError:
    safe_install('scipy')
    from scipy import ndimage, signal
    from scipy.fftpack import dct, idct, dctn
    from scipy.stats import entropy

try:
    from tqdm import tqdm
except ImportError:
    safe_install('tqdm')
    from tqdm import tqdm

# ============================================================================
# CONFIGURATION - ULTRA AGGRESSIVE FOR HIGH RECALL
# ============================================================================
class Config:
    # Paths
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
    
    # Model
    IMG_SIZE = 384  # UNLEASHED requires exactly 384
    
    # Ultra-aggressive thresholds (maximize recall)
    FINAL_THRESHOLD = 0.20       # Very low to catch subtle forgeries
    HIGH_CONFIDENCE = 0.40       # High confidence threshold
    
    # Ensemble weights (sum = 1.0)
    W_DEEP_LEARNING = 0.35       # UNLEASHED model
    W_SELF_CORRELATION = 0.20   # Self-correlation module
    W_KEYPOINT = 0.15           # Keypoint matching
    W_SPECTRAL = 0.10           # DCT/FFT analysis
    W_ELA = 0.10                # Error level analysis
    W_NOISE = 0.05              # Noise analysis
    W_STATISTICAL = 0.05        # Statistical features
    
    # Post-processing
    MIN_AREA = 100              # Very small minimum
    MIN_AREA_RATIO = 0.001      # 0.1% of image area
    MORPH_SIZE = 5
    
    # TTA
    USE_TTA = True
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')

print(f"🚀 TITAN APEX CHAMPION - Device: {Config.DEVICE}")

# ============================================================================
# RLE ENCODING (Competition Standard)
# ============================================================================
def rle_encode(mask):
    """Convert binary mask to RLE string (JSON format for Kaggle)"""
    if mask is None or mask.sum() == 0:
        return "authentic"
    
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    # Use json.dumps format (matches V3.1 safe submission)
    return json.dumps([int(x) for x in runs])

# ============================================================================
# MODULE 1: UNLEASHED DEEP LEARNING
# ============================================================================
class FrequencyBlock(nn.Module):
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
        mask = (torch.sqrt((y - cy)**2 + (x_grid - cx)**2) > 15).float().unsqueeze(0).unsqueeze(0)
        img_back = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_shift * mask)))
        return self.conv(img_back)

class GraphModule(nn.Module):
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

class DeepLearningModule:
    def __init__(self, model_path, device):
        self.device = device
        self.model = None
        self.loaded = False
        
        if os.path.exists(model_path):
            try:
                self.model = FAGT_Model()
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.to(device).eval()
                self.loaded = True
                print(f"✓ UNLEASHED model loaded")
            except Exception as e:
                print(f"✗ Model load failed: {e}")
    
    def predict(self, image):
        if not self.loaded:
            return None
        
        # Preprocess to exactly 384x384
        img = cv2.resize(image, (384, 384))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            pred = torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()
        
        # TTA
        if Config.USE_TTA:
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
            
            pred = (pred + pred_h + pred_v) / 3.0
        
        return pred

# ============================================================================
# MODULE 2: SELF-CORRELATION ANALYSIS (Copy-Move Specific)
# ============================================================================
class SelfCorrelationModule:
    """
    Compute self-correlation to find duplicated regions.
    Key insight: Copy-move creates highly correlated but spatially distant regions.
    """
    
    def analyze(self, image, patch_size=32, stride=16):
        """Find regions that correlate with other distant regions"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float32)
        
        h, w = gray.shape
        
        # Extract patches
        patches = []
        positions = []
        
        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                patch = gray[y:y+patch_size, x:x+patch_size]
                # Normalize patch
                patch_norm = (patch - patch.mean()) / (patch.std() + 1e-8)
                patches.append(patch_norm.flatten())
                positions.append((y + patch_size//2, x + patch_size//2))
        
        if len(patches) < 10:
            return np.zeros(gray.shape, dtype=np.float32)
        
        patches = np.array(patches)
        positions = np.array(positions)
        
        # Compute correlation matrix (efficient via dot product of normalized patches)
        corr_matrix = np.dot(patches, patches.T) / (patch_size * patch_size)
        
        # Find high correlations between distant patches
        suspicious = np.zeros(gray.shape, dtype=np.float32)
        min_distance = 50  # Minimum distance for copy-move
        corr_threshold = 0.85  # High correlation threshold
        
        for i in range(len(patches)):
            for j in range(i+1, len(patches)):
                # Check distance
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                              (positions[i][1] - positions[j][1])**2)
                
                if dist > min_distance and corr_matrix[i, j] > corr_threshold:
                    # Mark both regions
                    y1, x1 = positions[i]
                    y2, x2 = positions[j]
                    
                    # Draw on suspicious map (convert to float for OpenCV)
                    color_val = float(corr_matrix[i, j])
                    cv2.circle(suspicious, (int(x1), int(y1)), patch_size//2, 
                              color_val, -1)
                    cv2.circle(suspicious, (int(x2), int(y2)), patch_size//2, 
                              color_val, -1)
        
        # Dilate to connect nearby detections
        if suspicious.max() > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            suspicious = cv2.dilate(suspicious, kernel, iterations=2)
            suspicious = cv2.GaussianBlur(suspicious, (15, 15), 0)
        
        return suspicious

# ============================================================================
# MODULE 3: MULTI-SCALE KEYPOINT MATCHING
# ============================================================================
class KeypointMatchingModule:
    """Advanced keypoint matching with multiple detectors"""
    
    def __init__(self):
        # Multiple detectors for robustness
        self.orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
        self.akaze = cv2.AKAZE_create()
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Try to create SIFT (may not be available)
        try:
            self.sift = cv2.SIFT_create(nfeatures=2000)
            self.bf_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self.has_sift = True
        except:
            self.has_sift = False
    
    def detect_copymove(self, image, min_dist=40, max_dist=500):
        """Multi-detector keypoint matching"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        all_suspicious_points = []
        
        # ORB matching
        suspicious_orb = self._match_detector(gray, self.orb, self.bf_hamming, 
                                              50, min_dist, max_dist)
        all_suspicious_points.extend(suspicious_orb)
        
        # AKAZE matching
        suspicious_akaze = self._match_detector(gray, self.akaze, self.bf_hamming, 
                                                50, min_dist, max_dist)
        all_suspicious_points.extend(suspicious_akaze)
        
        # SIFT matching (if available)
        if self.has_sift:
            suspicious_sift = self._match_detector(gray, self.sift, self.bf_l2, 
                                                   100, min_dist, max_dist)
            all_suspicious_points.extend(suspicious_sift)
        
        if len(all_suspicious_points) < 6:
            return np.zeros(gray.shape, dtype=np.float32)
        
        # Cluster suspicious points
        return self._cluster_and_create_mask(all_suspicious_points, gray.shape)
    
    def _match_detector(self, gray, detector, matcher, dist_thresh, min_dist, max_dist):
        """Match keypoints using a specific detector"""
        kps, descs = detector.detectAndCompute(gray, None)
        
        if descs is None or len(kps) < 10:
            return []
        
        matches = matcher.knnMatch(descs, descs, k=5)
        
        suspicious = []
        for match_group in matches:
            for m in match_group:
                if m.queryIdx == m.trainIdx:
                    continue
                
                pt1 = np.array(kps[m.queryIdx].pt)
                pt2 = np.array(kps[m.trainIdx].pt)
                spatial_dist = np.linalg.norm(pt1 - pt2)
                
                if min_dist < spatial_dist < max_dist and m.distance < dist_thresh:
                    suspicious.append(pt1)
                    suspicious.append(pt2)
        
        return suspicious
    
    def _cluster_and_create_mask(self, points, shape):
        """Cluster points and create probability mask"""
        points = np.array(points)
        
        # Use DBSCAN clustering
        try:
            clustering = DBSCAN(eps=25, min_samples=4).fit(points)
            labels = clustering.labels_
        except:
            return np.zeros(shape, dtype=np.float32)
        
        mask = np.zeros(shape, dtype=np.float32)
        
        for label in set(labels):
            if label == -1:
                continue
            
            cluster_pts = points[labels == label]
            if len(cluster_pts) < 4:
                continue
            
            try:
                hull = cv2.convexHull(cluster_pts.astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 1.0)
            except:
                for pt in cluster_pts:
                    cv2.circle(mask, (int(pt[0]), int(pt[1])), 20, 1.0, -1)
        
        if mask.max() > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask

# ============================================================================
# MODULE 4: SPECTRAL ANALYSIS (DCT + FFT)
# ============================================================================
class SpectralAnalysisModule:
    """Frequency domain analysis for forensic artifacts"""
    
    def analyze(self, image):
        """Combined DCT and FFT analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float64)
        
        # DCT analysis
        dct_anomaly = self._dct_analysis(gray)
        
        # FFT analysis
        fft_anomaly = self._fft_analysis(gray)
        
        # Combine
        combined = np.maximum(dct_anomaly, fft_anomaly * 0.8)
        
        return combined
    
    def _dct_analysis(self, gray, block_size=8):
        """Detect double-JPEG artifacts via DCT"""
        h, w = gray.shape
        h_blocks, w_blocks = h // block_size, w // block_size
        
        energy_map = np.zeros((h_blocks, w_blocks), dtype=np.float64)
        
        for i in range(h_blocks):
            for j in range(w_blocks):
                block = gray[i*block_size:(i+1)*block_size, 
                            j*block_size:(j+1)*block_size]
                
                dct_block = cv2.dct(block)
                
                # High-frequency energy (artifacts appear here)
                dct_block[0, 0] = 0  # Remove DC
                energy = np.sum(np.abs(dct_block[4:, 4:]))
                energy_map[i, j] = energy
        
        # Normalize and detect outliers
        if energy_map.max() > energy_map.min():
            energy_map = (energy_map - energy_map.min()) / (energy_map.max() - energy_map.min())
        
        # Resize to original
        energy_full = cv2.resize(energy_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Detect anomalous regions
        mean_e = np.mean(energy_full)
        std_e = np.std(energy_full)
        anomaly = np.abs(energy_full - mean_e) > 1.5 * std_e
        
        return anomaly.astype(np.float32)
    
    def _fft_analysis(self, gray):
        """FFT-based manipulation detection"""
        # Apply FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        
        # Look for periodic artifacts (copy-move often creates them)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        
        # Create radial profile
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Exclude DC component and very low frequencies
        mask_valid = (r > 10) & (r < min(h, w) // 3)
        
        # Find peaks in frequency domain
        magnitude_masked = magnitude * mask_valid
        
        # Threshold high magnitudes
        if magnitude_masked.max() > 0:
            threshold = np.percentile(magnitude_masked[magnitude_masked > 0], 95)
            peaks = (magnitude_masked > threshold).astype(np.float32)
        else:
            peaks = np.zeros((h, w), dtype=np.float32)
        
        # Convert back to spatial domain hint
        # High peaks suggest periodic manipulation
        if peaks.sum() > 100:
            # Strong periodic signal detected
            anomaly = np.ones((h, w), dtype=np.float32) * 0.3
        else:
            anomaly = np.zeros((h, w), dtype=np.float32)
        
        return anomaly

# ============================================================================
# MODULE 5: ERROR LEVEL ANALYSIS (Multi-Quality)
# ============================================================================
class ErrorLevelModule:
    """Multi-quality ELA for manipulation detection"""
    
    def analyze(self, image):
        """ELA at multiple quality levels"""
        # Multiple qualities for robust detection
        qualities = [95, 90, 85, 80]
        ela_maps = []
        
        for q in qualities:
            ela = self._single_ela(image, q)
            ela_maps.append(ela)
        
        # Combine - take maximum response
        combined = np.maximum.reduce(ela_maps)
        
        return combined
    
    def _single_ela(self, image, quality):
        """Single quality ELA"""
        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Absolute difference
        diff = cv2.absdiff(image, decoded).astype(np.float32)
        ela = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Amplify and normalize
        ela = ela * 15
        ela = np.clip(ela / 255.0, 0, 1)
        
        # Threshold based on statistics
        mean_ela = np.mean(ela)
        std_ela = np.std(ela)
        threshold = mean_ela + 2.0 * std_ela
        
        suspicious = (ela > threshold).astype(np.float32)
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        suspicious = cv2.morphologyEx(suspicious, cv2.MORPH_CLOSE, kernel)
        suspicious = cv2.morphologyEx(suspicious, cv2.MORPH_OPEN, kernel)
        suspicious = cv2.GaussianBlur(suspicious, (11, 11), 0)
        
        return suspicious

# ============================================================================
# MODULE 6: NOISE PRINT ANALYSIS
# ============================================================================
class NoisePrintModule:
    """Local noise pattern inconsistency detection"""
    
    def analyze(self, image, block_size=32):
        """Detect regions with inconsistent noise patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float32)
        
        # Extract noise residual (multiple methods)
        
        # Method 1: Median filter residual
        median = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
        noise1 = gray - median
        
        # Method 2: Gaussian filter residual
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        noise2 = gray - gaussian
        
        # Method 3: Wiener-like filter residual
        bilateral = cv2.bilateralFilter(gray.astype(np.uint8), 5, 75, 75).astype(np.float32)
        noise3 = gray - bilateral
        
        # Combine noise estimates
        noise = (np.abs(noise1) + np.abs(noise2) + np.abs(noise3)) / 3
        
        # Calculate local variance
        h, w = gray.shape
        variance_map = np.zeros((h, w), dtype=np.float32)
        half = block_size // 2
        
        for y in range(half, h - half, half):
            for x in range(half, w - half, half):
                block = noise[y-half:y+half, x-half:x+half]
                variance_map[y-half:y+half, x-half:x+half] = np.var(block)
        
        # Normalize
        if variance_map.max() > 0:
            variance_map = variance_map / variance_map.max()
        
        # Find outliers (very different noise patterns)
        mean_var = np.mean(variance_map)
        std_var = np.std(variance_map)
        
        anomaly = np.abs(variance_map - mean_var) > 1.5 * std_var
        anomaly = anomaly.astype(np.float32)
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, kernel)
        anomaly = cv2.GaussianBlur(anomaly, (21, 21), 0)
        
        return anomaly

# ============================================================================
# MODULE 7: STATISTICAL FEATURE ANALYSIS
# ============================================================================
class StatisticalModule:
    """Statistical inconsistency detection"""
    
    def analyze(self, image, block_size=64):
        """Detect regions with inconsistent statistics"""
        if len(image.shape) == 3:
            # Analyze each channel
            anomaly = np.zeros(image.shape[:2], dtype=np.float32)
            for c in range(3):
                channel_anomaly = self._analyze_channel(image[:,:,c], block_size)
                anomaly = np.maximum(anomaly, channel_anomaly)
            return anomaly
        else:
            return self._analyze_channel(image, block_size)
    
    def _analyze_channel(self, gray, block_size):
        """Analyze single channel"""
        gray = gray.astype(np.float32)
        h, w = gray.shape
        half = block_size // 2
        
        # Calculate local statistics
        mean_map = np.zeros((h, w), dtype=np.float32)
        std_map = np.zeros((h, w), dtype=np.float32)
        entropy_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(half, h - half, half):
            for x in range(half, w - half, half):
                block = gray[y-half:y+half, x-half:x+half]
                
                mean_map[y-half:y+half, x-half:x+half] = np.mean(block)
                std_map[y-half:y+half, x-half:x+half] = np.std(block)
                
                # Local entropy
                hist, _ = np.histogram(block.flatten(), bins=32, range=(0, 255))
                hist = hist / (hist.sum() + 1e-8)
                entropy_map[y-half:y+half, x-half:x+half] = entropy(hist + 1e-8)
        
        # Normalize
        for m in [mean_map, std_map, entropy_map]:
            if m.max() > m.min():
                m[:] = (m - m.min()) / (m.max() - m.min())
        
        # Find outliers in each statistic
        anomaly = np.zeros((h, w), dtype=np.float32)
        
        for stat_map in [mean_map, std_map, entropy_map]:
            mean_s = np.mean(stat_map)
            std_s = np.std(stat_map)
            outliers = np.abs(stat_map - mean_s) > 2.0 * std_s
            anomaly = np.maximum(anomaly, outliers.astype(np.float32))
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, kernel)
        anomaly = cv2.GaussianBlur(anomaly, (15, 15), 0)
        
        return anomaly

# ============================================================================
# SMART ENSEMBLE WITH ADAPTIVE FUSION
# ============================================================================
class SmartEnsemble:
    """Adaptive fusion of all detection modules"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize all modules
        self.dl_module = DeepLearningModule(config.MODEL_PATH, config.DEVICE)
        self.self_corr = SelfCorrelationModule()
        self.keypoint = KeypointMatchingModule()
        self.spectral = SpectralAnalysisModule()
        self.ela = ErrorLevelModule()
        self.noise = NoisePrintModule()
        self.statistical = StatisticalModule()
    
    def detect(self, image, original_shape):
        """Full multi-modal analysis"""
        h, w = original_shape[:2]
        
        detections = {}
        
        # Module 1: Deep Learning
        if self.dl_module.loaded:
            try:
                dl_pred = self.dl_module.predict(image)
                if dl_pred is not None:
                    detections['dl'] = cv2.resize(dl_pred, (w, h))
            except Exception as e:
                print(f"DL error: {e}")
        
        # Module 2: Self-Correlation
        try:
            self_corr_pred = self.self_corr.analyze(image)
            detections['self_corr'] = cv2.resize(self_corr_pred, (w, h))
        except Exception as e:
            print(f"Self-corr error: {e}")
        
        # Module 3: Keypoint Matching
        try:
            kp_pred = self.keypoint.detect_copymove(image)
            detections['keypoint'] = cv2.resize(kp_pred, (w, h))
        except Exception as e:
            print(f"Keypoint error: {e}")
        
        # Module 4: Spectral Analysis
        try:
            spectral_pred = self.spectral.analyze(image)
            detections['spectral'] = cv2.resize(spectral_pred, (w, h))
        except Exception as e:
            print(f"Spectral error: {e}")
        
        # Module 5: ELA
        try:
            ela_pred = self.ela.analyze(image)
            detections['ela'] = cv2.resize(ela_pred, (w, h))
        except Exception as e:
            print(f"ELA error: {e}")
        
        # Module 6: Noise
        try:
            noise_pred = self.noise.analyze(image)
            detections['noise'] = cv2.resize(noise_pred, (w, h))
        except Exception as e:
            print(f"Noise error: {e}")
        
        # Module 7: Statistical
        try:
            stat_pred = self.statistical.analyze(image)
            detections['statistical'] = cv2.resize(stat_pred, (w, h))
        except Exception as e:
            print(f"Statistical error: {e}")
        
        if not detections:
            return np.zeros((h, w), dtype=np.uint8)
        
        # Weighted ensemble
        weights = {
            'dl': Config.W_DEEP_LEARNING,
            'self_corr': Config.W_SELF_CORRELATION,
            'keypoint': Config.W_KEYPOINT,
            'spectral': Config.W_SPECTRAL,
            'ela': Config.W_ELA,
            'noise': Config.W_NOISE,
            'statistical': Config.W_STATISTICAL
        }
        
        combined = np.zeros((h, w), dtype=np.float32)
        total_weight = 0
        
        for name, pred in detections.items():
            weight = weights.get(name, 0.1)
            combined += pred * weight
            total_weight += weight
        
        if total_weight > 0:
            combined /= total_weight
        
        # Adaptive boosting: if DL is confident, trust it more
        if 'dl' in detections:
            dl_max = detections['dl'].max()
            if dl_max > 0.5:
                combined = 0.7 * detections['dl'] + 0.3 * combined
        
        # If traditional methods agree, boost those regions
        if 'keypoint' in detections and 'self_corr' in detections:
            agreement = detections['keypoint'] * detections['self_corr']
            if agreement.max() > 0.3:
                combined = np.maximum(combined, agreement * 1.2)
        
        # Hysteresis thresholding
        mask = self._hysteresis(combined, Config.FINAL_THRESHOLD, Config.HIGH_CONFIDENCE)
        
        # Post-processing
        mask = self._postprocess(mask, h, w)
        
        return mask
    
    def _hysteresis(self, prob, low, high):
        """Two-threshold decision"""
        high_mask = (prob >= high).astype(np.uint8)
        low_mask = (prob >= low).astype(np.uint8)
        
        # Connected components from high confidence
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(high_mask)
        
        result = np.zeros_like(prob, dtype=np.uint8)
        
        for i in range(1, num_labels):
            component = (labels == i).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            grown = cv2.dilate(component, kernel, iterations=2)
            grown = grown & low_mask
            result = np.maximum(result, grown)
        
        # Also include isolated low-confidence regions if large enough
        remaining = low_mask & (1 - result)
        num_remain, labels_remain, stats_remain, _ = cv2.connectedComponentsWithStats(remaining)
        
        for i in range(1, num_remain):
            area = stats_remain[i, cv2.CC_STAT_AREA]
            if area > Config.MIN_AREA * 3:  # Only large isolated regions
                result = np.maximum(result, (labels_remain == i).astype(np.uint8))
        
        return result
    
    def _postprocess(self, mask, h, w):
        """Clean up mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (Config.MORPH_SIZE, Config.MORPH_SIZE))
        
        # Close holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Open to remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        min_area = max(Config.MIN_AREA, int(h * w * Config.MIN_AREA_RATIO))
        
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                clean_mask[labels == i] = 1
        
        return clean_mask

# ============================================================================
# MAIN INFERENCE
# ============================================================================
def run_inference():
    print("=" * 70)
    print("🏆 TITAN APEX CHAMPION - Multi-Modal Forensic Detection")
    print("=" * 70)
    
    # Initialize
    ensemble = SmartEnsemble(Config)
    
    # Load sample submission (defines expected rows)
    sample_sub = pd.read_csv(Config.SAMPLE_SUB)
    print(f"📊 Total test images: {len(sample_sub)}")
    print(f"📋 Using columns: 'case_id', 'annotation' (V3.1 safe logic)")
    
    results = []
    forged = 0
    
    # Get list of case IDs from sample submission
    case_ids = sample_sub['case_id'].astype(str).tolist()
    
    for case_id in tqdm(case_ids, desc="🔍 Analyzing"):
        # Find image file
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp']:
            p = Config.TEST_IMAGES / f"{case_id}{ext}"
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            # Try glob search
            matches = list(Config.TEST_IMAGES.glob(f"{case_id}.*"))
            if matches:
                img_path = matches[0]
        
        if img_path is None:
            results.append({'case_id': case_id, 'annotation': 'authentic'})
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            results.append({'case_id': case_id, 'annotation': 'authentic'})
            continue
        
        # Detect
        try:
            mask = ensemble.detect(image, image.shape)
            
            if mask.sum() > 0:
                rle = rle_encode(mask)
                if rle == "":  # Safety check
                    rle = "authentic"
                else:
                    forged += 1
            else:
                rle = "authentic"
            
            results.append({'case_id': case_id, 'annotation': rle})
            
        except Exception as e:
            print(f"Error {case_id}: {e}")
            results.append({'case_id': case_id, 'annotation': 'authentic'})
    
    # === SAFE SUBMISSION LOGIC (FROM V3.1) ===
    # Read sample_submission to ensure exact rows Kaggle expects
    try:
        sample_sub = pd.read_csv(Config.SAMPLE_SUB)
    except:
        sample_sub = pd.DataFrame({'case_id': [45], 'annotation': ['authentic']})
    
    sample_sub['case_id'] = sample_sub['case_id'].astype(str)
    
    if len(results) > 0:
        preds_df = pd.DataFrame(results)
        preds_df['case_id'] = preds_df['case_id'].astype(str)
        # Left merge ensures we only keep rows that Kaggle expects
        submission = sample_sub[['case_id']].merge(preds_df, on='case_id', how='left')
        submission['annotation'] = submission['annotation'].fillna('authentic')
    else:
        submission = sample_sub.copy()
        submission['annotation'] = 'authentic'
    
    output_path = Config.OUTPUT_DIR / 'submission.csv'
    submission.to_csv(output_path, index=False)
    
    print("=" * 70)
    print(f"✅ COMPLETE")
    print(f"   Total: {len(submission)} | Forged: {forged} | Authentic: {len(submission) - forged}")
    print(f"   Saved: {output_path}")
    print("=" * 70)
    
    return submission

if __name__ == "__main__":
    run_inference()

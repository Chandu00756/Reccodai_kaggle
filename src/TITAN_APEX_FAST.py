"""
================================================================================
TITAN APEX FAST: Lightning-Speed 7-Module Forensic Detection
================================================================================
TARGET: 0.8+ F1 Score + <3 seconds per image

OPTIMIZATIONS:
├── GPU-accelerated DL with TTA
├── Downscaled analysis (512px max) for traditional modules
├── Vectorized numpy (no Python loops)
├── Early exit if DL confident
├── Efficient patch correlation via convolution
└── Safe V3.1 submission logic

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
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import timm
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm', '-q'], 
                  check=True, capture_output=True)
    import timm

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tqdm', '-q'], 
                  check=True, capture_output=True)
    from tqdm import tqdm

# ============================================================================
# CONFIGURATION
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
    
    # Model settings
    IMG_SIZE = 384
    ANALYSIS_SIZE = 512  # Downscale for fast traditional analysis
    
    # Thresholds (V3.1 style - strict)
    HIGH_THRESH = 0.85
    LOW_THRESH = 0.40
    MIN_PIXELS = 1000
    MIN_STRONG = 50
    
    # Ensemble weights
    W_DL = 0.50          # Trust DL most
    W_SELF_CORR = 0.15
    W_KEYPOINT = 0.10
    W_SPECTRAL = 0.08
    W_ELA = 0.08
    W_NOISE = 0.05
    W_STATS = 0.04
    
    # Speed settings
    DL_CONFIDENCE_SKIP = 0.90  # Skip traditional if DL > this
    USE_TTA = True
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')

print(f"⚡ TITAN APEX FAST - Device: {Config.DEVICE}")

# ============================================================================
# RLE ENCODING (V3.1 Format)
# ============================================================================
def rle_encode(mask):
    """JSON format RLE encoding"""
    if mask is None or mask.sum() == 0:
        return "authentic"
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return json.dumps([int(x) for x in runs])

# ============================================================================
# HYSTERESIS FILTERING (V3.1 Style)
# ============================================================================
def apply_hysteresis(prob_map, high, low, min_pixels, min_strong):
    """V3.1 strict filtering"""
    h, w = prob_map.shape
    strong_mask = (prob_map >= high).astype(np.uint8)
    weak_mask = (prob_map >= low).astype(np.uint8)
    
    if strong_mask.sum() < min_strong:
        return np.zeros((h, w), dtype=np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak_mask, connectivity=8)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_pixels:
            continue
        blob_mask = (labels == i).astype(np.uint8)
        strong_overlap = cv2.bitwise_and(blob_mask, strong_mask)
        if strong_overlap.sum() >= min_strong:
            final_mask = cv2.bitwise_or(final_mask, blob_mask)
    
    return final_mask

# ============================================================================
# MODULE 1: DEEP LEARNING (UNLEASHED)
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
                print(f"✓ UNLEASHED loaded on {device}")
            except Exception as e:
                print(f"✗ Model failed: {e}")
    
    @torch.no_grad()
    def predict(self, image):
        if not self.loaded:
            return None, 0.0
        
        # Preprocess
        img = cv2.resize(image, (384, 384))
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        
        # Forward + TTA
        pred = torch.sigmoid(self.model(tensor))[0, 0]
        
        if Config.USE_TTA:
            # Horizontal flip TTA
            pred_h = torch.sigmoid(self.model(torch.flip(tensor, [3])))[0, 0]
            pred_h = torch.flip(pred_h, [1])
            pred = (pred + pred_h) / 2.0
        
        pred_np = pred.cpu().numpy()
        confidence = float(pred_np.max())
        
        return pred_np, confidence

# ============================================================================
# MODULE 2: FAST SELF-CORRELATION (Vectorized)
# ============================================================================
def fast_self_correlation(gray, patch_size=48, stride=24):
    """Vectorized self-correlation using convolution trick"""
    h, w = gray.shape
    if h < patch_size or w < patch_size:
        return np.zeros((h, w), dtype=np.float32)
    
    # Extract patches efficiently
    patches = []
    positions = []
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = gray[y:y+patch_size, x:x+patch_size].flatten()
            patch = (patch - patch.mean()) / (patch.std() + 1e-8)
            patches.append(patch)
            positions.append((y + patch_size//2, x + patch_size//2))
    
    if len(patches) < 5:
        return np.zeros((h, w), dtype=np.float32)
    
    patches = np.array(patches)
    positions = np.array(positions)
    n_patches = len(patches)
    
    # Efficient correlation via matrix multiply
    corr = np.dot(patches, patches.T) / (patch_size * patch_size)
    
    # Find copy-move pairs (vectorized)
    suspicious = np.zeros((h, w), dtype=np.float32)
    
    # Distance matrix
    pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.sqrt((pos_diff ** 2).sum(axis=2))
    
    # Find high correlation + distant pairs
    mask = (distances > 60) & (corr > 0.80)
    pairs = np.where(mask)
    
    for i, j in zip(pairs[0], pairs[1]):
        if i >= j:
            continue
        y1, x1 = positions[i]
        y2, x2 = positions[j]
        cv2.circle(suspicious, (int(x1), int(y1)), patch_size//2, float(corr[i, j]), -1)
        cv2.circle(suspicious, (int(x2), int(y2)), patch_size//2, float(corr[i, j]), -1)
    
    if suspicious.max() > 0:
        suspicious = cv2.GaussianBlur(suspicious, (15, 15), 0)
    
    return suspicious

# ============================================================================
# MODULE 3: FAST KEYPOINT MATCHING
# ============================================================================
def fast_keypoint_matching(gray):
    """Fast ORB-only keypoint matching"""
    orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.3, nlevels=6)
    kps, descs = orb.detectAndCompute(gray, None)
    
    if descs is None or len(kps) < 10:
        return np.zeros(gray.shape, dtype=np.float32)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descs, descs, k=3)
    
    suspicious_pts = []
    for match_group in matches:
        for m in match_group:
            if m.queryIdx == m.trainIdx:
                continue
            pt1 = np.array(kps[m.queryIdx].pt)
            pt2 = np.array(kps[m.trainIdx].pt)
            dist = np.linalg.norm(pt1 - pt2)
            if 50 < dist < 400 and m.distance < 45:
                suspicious_pts.extend([pt1, pt2])
    
    if len(suspicious_pts) < 6:
        return np.zeros(gray.shape, dtype=np.float32)
    
    # Simple clustering via grid
    mask = np.zeros(gray.shape, dtype=np.float32)
    for pt in suspicious_pts:
        cv2.circle(mask, (int(pt[0]), int(pt[1])), 25, 1.0, -1)
    
    if mask.max() > 0:
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    return mask

# ============================================================================
# MODULE 4: FAST SPECTRAL (FFT only)
# ============================================================================
def fast_spectral(gray):
    """Fast FFT-based detection"""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    
    # Radial mask
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = (r > 15) & (r < min(h, w) // 3)
    
    mag_masked = magnitude * mask
    if mag_masked.max() > 0:
        threshold = np.percentile(mag_masked[mag_masked > 0], 97)
        peaks = (mag_masked > threshold).sum()
        if peaks > 50:
            return np.ones((h, w), dtype=np.float32) * 0.3
    
    return np.zeros((h, w), dtype=np.float32)

# ============================================================================
# MODULE 5: FAST ELA (Single quality)
# ============================================================================
def fast_ela(image):
    """Single-quality ELA"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, encoded = cv2.imencode('.jpg', image, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    diff = cv2.absdiff(image, decoded).astype(np.float32)
    ela = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ela = ela * 15 / 255.0
    
    mean_ela = np.mean(ela)
    std_ela = np.std(ela)
    suspicious = (ela > mean_ela + 2.5 * std_ela).astype(np.float32)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    suspicious = cv2.morphologyEx(suspicious, cv2.MORPH_CLOSE, kernel)
    suspicious = cv2.GaussianBlur(suspicious, (11, 11), 0)
    
    return suspicious

# ============================================================================
# MODULE 6: FAST NOISE ANALYSIS
# ============================================================================
def fast_noise(gray, block_size=48):
    """Fast noise variance analysis"""
    median = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    noise = np.abs(gray.astype(np.float32) - median)
    
    h, w = gray.shape
    bh, bw = h // block_size, w // block_size
    
    if bh < 2 or bw < 2:
        return np.zeros((h, w), dtype=np.float32)
    
    # Block variance (vectorized)
    noise_blocks = noise[:bh*block_size, :bw*block_size].reshape(bh, block_size, bw, block_size)
    variance = noise_blocks.var(axis=(1, 3))
    
    if variance.max() > variance.min():
        variance = (variance - variance.min()) / (variance.max() - variance.min())
    
    mean_v = variance.mean()
    std_v = variance.std()
    anomaly = (np.abs(variance - mean_v) > 2.0 * std_v).astype(np.float32)
    
    # Upscale
    anomaly = cv2.resize(anomaly, (w, h), interpolation=cv2.INTER_LINEAR)
    anomaly = cv2.GaussianBlur(anomaly, (21, 21), 0)
    
    return anomaly

# ============================================================================
# MODULE 7: FAST STATISTICAL
# ============================================================================
def fast_statistical(gray, block_size=64):
    """Fast block statistics"""
    h, w = gray.shape
    bh, bw = h // block_size, w // block_size
    
    if bh < 2 or bw < 2:
        return np.zeros((h, w), dtype=np.float32)
    
    gray_blocks = gray[:bh*block_size, :bw*block_size].reshape(bh, block_size, bw, block_size)
    
    means = gray_blocks.mean(axis=(1, 3))
    stds = gray_blocks.std(axis=(1, 3))
    
    # Normalize
    if means.max() > means.min():
        means = (means - means.min()) / (means.max() - means.min())
    if stds.max() > stds.min():
        stds = (stds - stds.min()) / (stds.max() - stds.min())
    
    # Find outliers
    anomaly = np.zeros_like(means)
    for stat in [means, stds]:
        m, s = stat.mean(), stat.std()
        outliers = np.abs(stat - m) > 2.5 * s
        anomaly = np.maximum(anomaly, outliers.astype(np.float32))
    
    # Upscale
    anomaly = cv2.resize(anomaly, (w, h), interpolation=cv2.INTER_LINEAR)
    anomaly = cv2.GaussianBlur(anomaly, (15, 15), 0)
    
    return anomaly

# ============================================================================
# FAST ENSEMBLE
# ============================================================================
class FastEnsemble:
    def __init__(self, config):
        self.config = config
        self.dl_module = DeepLearningModule(config.MODEL_PATH, config.DEVICE)
    
    def detect(self, image, original_shape):
        h, w = original_shape[:2]
        
        # 1. Deep Learning (always run)
        dl_pred, dl_conf = self.dl_module.predict(image)
        
        if dl_pred is None:
            return np.zeros((h, w), dtype=np.uint8)
        
        dl_pred_full = cv2.resize(dl_pred, (w, h))
        
        # Early exit if DL very confident it's authentic
        if dl_conf < 0.15:
            return np.zeros((h, w), dtype=np.uint8)
        
        # Early exit if DL very confident it's forged
        if dl_conf > Config.DL_CONFIDENCE_SKIP:
            mask = apply_hysteresis(dl_pred_full, Config.HIGH_THRESH, Config.LOW_THRESH,
                                   Config.MIN_PIXELS, Config.MIN_STRONG)
            return mask
        
        # 2. Run fast traditional modules on downscaled image
        scale = min(Config.ANALYSIS_SIZE / max(h, w), 1.0)
        if scale < 1.0:
            small = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            small = image
        
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        sh, sw = gray_small.shape
        
        # Run modules
        self_corr = fast_self_correlation(gray_small)
        keypoint = fast_keypoint_matching(gray_small)
        spectral = fast_spectral(gray_small)
        ela = fast_ela(small)
        noise = fast_noise(gray_small)
        stats = fast_statistical(gray_small)
        
        # Upscale all to original size
        self_corr = cv2.resize(self_corr, (w, h))
        keypoint = cv2.resize(keypoint, (w, h))
        spectral = cv2.resize(spectral, (w, h))
        ela = cv2.resize(ela, (w, h))
        noise = cv2.resize(noise, (w, h))
        stats = cv2.resize(stats, (w, h))
        
        # Weighted ensemble
        combined = (
            Config.W_DL * dl_pred_full +
            Config.W_SELF_CORR * self_corr +
            Config.W_KEYPOINT * keypoint +
            Config.W_SPECTRAL * spectral +
            Config.W_ELA * ela +
            Config.W_NOISE * noise +
            Config.W_STATS * stats
        )
        
        # Boost if traditional methods agree with DL
        agreement = dl_pred_full * (self_corr + keypoint) / 2
        if agreement.max() > 0.3:
            combined = np.maximum(combined, agreement * 1.1)
        
        # Apply V3.1 hysteresis
        mask = apply_hysteresis(combined, Config.HIGH_THRESH, Config.LOW_THRESH,
                               Config.MIN_PIXELS, Config.MIN_STRONG)
        
        return mask

# ============================================================================
# MAIN INFERENCE
# ============================================================================
def run_inference():
    print("=" * 70)
    print("⚡ TITAN APEX FAST - Lightning 7-Module Detection")
    print("=" * 70)
    
    # Initialize
    ensemble = FastEnsemble(Config)
    
    # Load sample submission
    sample_sub = pd.read_csv(Config.SAMPLE_SUB)
    print(f"📊 Total test images: {len(sample_sub)}")
    
    results = []
    forged = 0
    
    case_ids = sample_sub['case_id'].astype(str).tolist()
    
    for case_id in tqdm(case_ids, desc="⚡ Processing"):
        # Find image
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.tif', '.tiff', '.bmp']:
            p = Config.TEST_IMAGES / f"{case_id}{ext}"
            if p.exists():
                img_path = p
                break
        
        if img_path is None:
            matches = list(Config.TEST_IMAGES.glob(f"{case_id}.*"))
            if matches:
                img_path = matches[0]
        
        if img_path is None:
            results.append({'case_id': case_id, 'annotation': 'authentic'})
            continue
        
        # Load
        image = cv2.imread(str(img_path))
        if image is None:
            results.append({'case_id': case_id, 'annotation': 'authentic'})
            continue
        
        # Detect
        try:
            mask = ensemble.detect(image, image.shape)
            
            if mask.sum() > 0:
                rle = rle_encode(mask)
                if rle and rle != "":
                    forged += 1
                else:
                    rle = "authentic"
            else:
                rle = "authentic"
            
            results.append({'case_id': case_id, 'annotation': rle})
            
        except Exception as e:
            print(f"Error {case_id}: {e}")
            results.append({'case_id': case_id, 'annotation': 'authentic'})
    
    # === SAFE SUBMISSION LOGIC (V3.1) ===
    try:
        sample_sub = pd.read_csv(Config.SAMPLE_SUB)
    except:
        sample_sub = pd.DataFrame({'case_id': [45], 'annotation': ['authentic']})
    
    sample_sub['case_id'] = sample_sub['case_id'].astype(str)
    
    if len(results) > 0:
        preds_df = pd.DataFrame(results)
        preds_df['case_id'] = preds_df['case_id'].astype(str)
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

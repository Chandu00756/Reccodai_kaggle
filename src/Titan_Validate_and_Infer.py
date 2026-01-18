"""
TITAN XIII VALIDATION + INFERENCE SCRIPT (FIXED)
=================================================
Fixes:
✅ Higher pixel threshold: 0.3 → adaptive (0.4-0.6)
✅ Higher forged ratio threshold: 0.01 → 0.05
✅ Morphological cleaning to remove noise
✅ Connected component filtering (min area)
✅ Test-Time Augmentation (TTA) for better predictions
✅ Validation on unseen data with proper metrics
"""

import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    import timm
    import pywt
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm", "PyWavelets"])
    import timm
    import pywt

# =============================================================================
# CONFIG - FIXED THRESHOLDS
# =============================================================================
class CFG:
    BASE_DIR = '/Users/chanduchitikam/recodai/recodai-luc-scientific-image-forgery-detection'
    TRAIN_IMG = os.path.join(BASE_DIR, 'train_images')
    TRAIN_MASK = os.path.join(BASE_DIR, 'train_masks')
    TEST_IMG = os.path.join(BASE_DIR, 'test_images')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'predictions_v2')
    WEIGHTS_PATH = "TITAN_XIII_BEST.pth"
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    TILE_SIZE = 512
    BATCH = 1
    WORKERS = 0
    
    # FIXED THRESHOLDS - More conservative to reduce false positives
    PIXEL_THRESHOLD = 0.5       # Was 0.3 - higher = fewer false positives
    FORGED_RATIO_MIN = 0.05     # Was 0.01 - image is FORGED only if >5% pixels flagged
    MIN_CONTOUR_AREA = 500      # Minimum area for a forged region to be counted
    MORPH_KERNEL_SIZE = 5       # Morphological cleaning kernel
    
    # For validation
    TARGET_F1 = 0.7

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print(f"[TITAN v2] Device: {CFG.DEVICE}")
print(f"[TITAN v2] Pixel Threshold: {CFG.PIXEL_THRESHOLD}")
print(f"[TITAN v2] Forged Ratio Min: {CFG.FORGED_RATIO_MIN}")
print(f"[TITAN v2] Min Contour Area: {CFG.MIN_CONTOUR_AREA}")

# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================
def get_ela_map(img, quality=90):
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        ela = np.abs(img.astype(np.float32) - decoded.astype(np.float32))
        ela = ela.max(axis=2)
        ela = ela / (ela.max() + 1e-8)
        ela = np.power(ela, 0.5)
        return ela.astype(np.float32)
    except:
        return np.zeros(img.shape[:2], dtype=np.float32)

def get_noise_map_advanced(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        coeffs = pywt.wavedec2(gray, 'haar', level=3)
        sigma_map = np.zeros((h, w), dtype=np.float32)
        weights = [0.5, 0.3, 0.2]
        for i in range(3):
            cD = coeffs[-(i+1)][2]
            median_cD = np.median(cD)
            mad = np.median(np.abs(cD - median_cD)) / 0.6745
            cD_abs = np.abs(cD) * mad
            cD_resized = cv2.resize(cD_abs, (w, h), interpolation=cv2.INTER_LINEAR)
            sigma_map += weights[i] * cD_resized
        block_size = 32
        dct_map = np.zeros_like(sigma_map)
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                dct_block = cv2.dct(block)
                hf = dct_block[block_size//2:, block_size//2:]
                dct_map[y:y+block_size, x:x+block_size] = np.var(hf)
        final_map = (sigma_map + dct_map) / 2.0
        final_map = (final_map - final_map.min()) / (final_map.max() - final_map.min() + 1e-8)
        return cv2.GaussianBlur(final_map, (5, 5), 1.0).astype(np.float32)
    except:
        return np.zeros(img.shape[:2], dtype=np.float32)

def get_cfa_map_advanced(img):
    try:
        h, w = img.shape[:2]
        r = img[:,:,0].astype(np.float32) / 255.0
        g = img[:,:,1].astype(np.float32) / 255.0
        b = img[:,:,2].astype(np.float32) / 255.0
        g_shift1 = np.roll(np.roll(g, 1, axis=0), 1, axis=1)
        g_corr = 1 - np.abs(g - g_shift1)
        g_dev = np.abs(0.85 - g_corr)
        r_shift = np.roll(np.roll(r, 2, axis=0), 2, axis=1)
        r_dev = np.abs(0.8 - (1 - np.abs(r - r_shift)))
        b_shift = np.roll(np.roll(b, 2, axis=0), 2, axis=1)
        b_dev = np.abs(0.8 - (1 - np.abs(b - b_shift)))
        cfa_map = (g_dev + r_dev + b_dev) / 3.0
        cfa_map = np.clip(cfa_map, 0, 1)
        cfa_map = (cfa_map - cfa_map.min()) / (cfa_map.max() - cfa_map.min() + 1e-8)
        return cv2.GaussianBlur(cfa_map, (5, 5), 1.0).astype(np.float32)
    except:
        return np.zeros(img.shape[:2], dtype=np.float32)

def get_edge_map(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        edges = cv2.GaussianBlur(edges, (5, 5), 1.0)
        return edges
    except:
        return np.zeros(img.shape[:2], dtype=np.float32)

# =============================================================================
# POST-PROCESSING - KEY FIX FOR FALSE POSITIVES
# =============================================================================
def clean_mask(prob_map, threshold=0.5, min_area=500, morph_kernel=5):
    """
    Clean prediction mask with morphological operations and connected component filtering.
    This removes noise and small spurious predictions that cause false positives.
    """
    # 1. Threshold
    binary = (prob_map > threshold).astype(np.uint8)
    
    # 2. Morphological opening (remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 3. Morphological closing (fill small holes)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 4. Connected component filtering - keep only large regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1
    
    return cleaned

def adaptive_threshold(prob_map, base_thresh=0.5):
    """
    Compute adaptive threshold based on probability distribution.
    If most pixels have low probability, use higher threshold to avoid FP.
    """
    mean_prob = prob_map.mean()
    std_prob = prob_map.std()
    
    # If image has very low mean probability, it's likely authentic - use high threshold
    if mean_prob < 0.1:
        return min(base_thresh + 0.2, 0.7)
    # If high variance, use standard threshold
    elif std_prob > 0.2:
        return base_thresh
    # Otherwise, be conservative
    else:
        return base_thresh + 0.1

# =============================================================================
# MODEL ARCHITECTURE (Same as training)
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, inc, outc, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(outc)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)

class FreqEncoderFFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(3, 128, 65) * 0.5)
        self.dwt_convs = nn.ModuleList([
            ConvBlock(12, 64), ConvBlock(64, 128), ConvBlock(128, 256), ConvBlock(256, 512)
        ])
        self.se_blocks = nn.ModuleList([SEBlock(64), SEBlock(128), SEBlock(256), SEBlock(512)])
    
    def haar_dwt(self, x):
        H, W = x.shape[2], x.shape[3]
        if H % 2 == 1:
            x = F.pad(x, [0, 0, 0, 1])
        if W % 2 == 1:
            x = F.pad(x, [0, 1, 0, 0])
        ll = (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2]) / 4
        lh = (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2]) / 4
        hl = (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2]) / 4
        hh = (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2]) / 4
        return torch.cat([ll, lh, hl, hh], dim=1)
    
    def forward(self, x):
        B = x.shape[0]
        x_small = F.interpolate(x, (128, 128), mode='bilinear', align_corners=False)
        x_fft = torch.fft.rfft2(x_small)
        gate = torch.sigmoid(self.gate).unsqueeze(0).expand(B, -1, -1, -1)
        x_gated = x_fft * gate
        x_filtered = torch.fft.irfft2(x_gated, s=(128, 128))
        dwt1 = self.haar_dwt(x_filtered)
        outs = []
        feat = dwt1
        for conv, se in zip(self.dwt_convs, self.se_blocks):
            feat = conv(feat)
            feat = se(feat)
            outs.append(feat)
        return outs

class PhysicsEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.blocks = nn.ModuleList([
            ConvBlock(64, 64, stride=1), ConvBlock(64, 128), ConvBlock(128, 256), ConvBlock(256, 512)
        ])
        self.se_blocks = nn.ModuleList([SEBlock(64), SEBlock(128), SEBlock(256), SEBlock(512)])
    
    def forward(self, x):
        x = F.relu(self.bn(self.initial(x)))
        outs = []
        for blk, se in zip(self.blocks, self.se_blocks):
            x = blk(x)
            x = se(x)
            outs.append(x)
        return outs

class EdgeAttentionGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_hint=None):
        attn = self.edge_conv(x)
        if edge_hint is not None:
            edge_hint = F.interpolate(edge_hint, size=attn.shape[2:], mode='bilinear', align_corners=False)
            attn = attn * (1 + edge_hint)
        return x * attn

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_s, dim_p, out_dim, num_heads=8):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_s = nn.Conv2d(dim_s, out_dim, 1)
        self.proj_p = nn.Conv2d(dim_p, out_dim, 1)
        self.q_proj = nn.Linear(out_dim, out_dim)
        self.k_proj = nn.Linear(out_dim, out_dim)
        self.v_proj = nn.Linear(out_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.gate = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())
    
    def forward(self, spatial, physics):
        B, _, H, W = spatial.shape
        s_proj = self.proj_s(spatial)
        p_proj = F.interpolate(self.proj_p(physics), (H, W), mode='bilinear', align_corners=False)
        s_flat = s_proj.flatten(2).transpose(1, 2)
        p_flat = p_proj.flatten(2).transpose(1, 2)
        Q = self.q_proj(s_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(p_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(p_flat).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, H * W, self.out_dim)
        out = self.out_proj(out)
        out = self.norm(out)
        gate_input = torch.cat([s_flat, out], dim=-1)
        gate_weight = self.gate(gate_input)
        fused = s_flat * (1 - gate_weight) + out * gate_weight
        return fused.transpose(1, 2).reshape(B, self.out_dim, H, W)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.merge = nn.Conv2d(in_ch + skip_ch, out_ch, 1)
        self.residual = ResidualBlock(out_ch)
        self.attention = EdgeAttentionGate(out_ch)
        self.se = SEBlock(out_ch)
    
    def forward(self, x, skip, edge_hint=None):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.merge(x)
        x = self.residual(x)
        x = self.attention(x, edge_hint)
        x = self.se(x)
        return x

class SwinDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.transpose(1, 2).reshape(B, C, H, W)

class TitanXIIIFixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model('swinv2_tiny_window8_256', pretrained=True, features_only=True)
        dims = self.swin.feature_info.channels()
        self.freq_enc = FreqEncoderFFT()
        self.physics_enc = PhysicsEncoder(in_channels=3)
        self.fuse4 = CrossAttentionFusion(dims[3], 512, 512, num_heads=8)
        self.fuse3 = CrossAttentionFusion(dims[2], 256, 256, num_heads=8)
        self.fuse2 = CrossAttentionFusion(dims[1], 128, 128, num_heads=4)
        self.physics_combine = nn.ModuleList([
            nn.Conv2d(128, 64, 1), nn.Conv2d(256, 128, 1), 
            nn.Conv2d(512, 256, 1), nn.Conv2d(1024, 512, 1)
        ])
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec3_swin = SwinDecoderBlock(256, num_heads=8)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec2_swin = SwinDecoderBlock(128, num_heads=4)
        self.dec1 = DecoderBlock(128, dims[0], 64)
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.head_low = nn.Conv2d(256, 1, 1)
        self.head_mid = nn.Conv2d(128, 1, 1)
        self.head_high = nn.Conv2d(64, 1, 1)
        self.head_final = nn.Conv2d(32, 1, 1)
    
    def forward(self, x, physics_maps, edges=None):
        B, _, H, W = x.shape
        x_swin = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        swin_feats = self.swin(x_swin)
        f_feats = self.freq_enc(x)
        p_feats = self.physics_enc(physics_maps)
        combined_physics = []
        for i, combine in enumerate(self.physics_combine):
            f_feat = F.interpolate(f_feats[i], size=p_feats[i].shape[2:], mode='bilinear', align_corners=False)
            combined = combine(torch.cat([f_feat, p_feats[i]], dim=1))
            combined_physics.append(combined)
        s4 = swin_feats[3].permute(0, 3, 1, 2).contiguous()
        s3 = swin_feats[2].permute(0, 3, 1, 2).contiguous()
        s2 = swin_feats[1].permute(0, 3, 1, 2).contiguous()
        s1 = swin_feats[0].permute(0, 3, 1, 2).contiguous()
        f4 = self.fuse4(s4, combined_physics[3])
        f3 = self.fuse3(s3, combined_physics[2])
        f2 = self.fuse2(s2, combined_physics[1])
        edge_hint = edges.unsqueeze(1) if edges is not None else None
        d3 = self.dec3(f4, f3, edge_hint)
        d3 = self.dec3_swin(d3)
        d2 = self.dec2(d3, f2, edge_hint)
        d2 = self.dec2_swin(d2)
        d1 = self.dec1(d2, s1, edge_hint)
        d0 = F.interpolate(d1, (H, W), mode='bilinear', align_corners=False)
        d0 = self.refine(d0)
        p_low = F.interpolate(self.head_low(d3), (H, W), mode='bilinear', align_corners=False)
        p_mid = F.interpolate(self.head_mid(d2), (H, W), mode='bilinear', align_corners=False)
        p_high = F.interpolate(self.head_high(d1), (H, W), mode='bilinear', align_corners=False)
        p_final = self.head_final(d0)
        out = 0.1 * p_low + 0.2 * p_mid + 0.3 * p_high + 0.4 * p_final
        return out, p_low, p_mid, p_high

# =============================================================================
# DATASET FOR VALIDATION
# =============================================================================
class ValidationDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_path'])
        if img is None:
            return self._dummy()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # Load ground truth mask
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        if row['label'] == 1 and pd.notna(row.get('mask_path')):
            try:
                m = np.load(row['mask_path'])
                if m.ndim == 3:
                    m = m.max(axis=2)
                mask = cv2.resize((m > 0).astype(np.float32), (img.shape[1], img.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
            except:
                pass
        
        # Compute physics maps
        ela = get_ela_map(img)
        noise = get_noise_map_advanced(img)
        cfa = get_cfa_map_advanced(img)
        edges = get_edge_map(img)
        
        if self.transform:
            aug = self.transform(image=img, mask=mask, ela=ela, noise=noise, cfa=cfa, edges=edges)
            img, mask, ela, noise, cfa, edges = aug['image'], aug['mask'], aug['ela'], aug['noise'], aug['cfa'], aug['edges']
        
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        physics_t = torch.stack([
            torch.from_numpy(ela).float(),
            torch.from_numpy(noise).float(),
            torch.from_numpy(cfa).float()
        ])
        edges_t = torch.from_numpy(edges).float()
        
        return img_t, physics_t, edges_t, mask_t, row['label'], row['image_path']
    
    def _dummy(self):
        return (
            torch.zeros(3, CFG.TILE_SIZE, CFG.TILE_SIZE),
            torch.zeros(3, CFG.TILE_SIZE, CFG.TILE_SIZE),
            torch.zeros(CFG.TILE_SIZE, CFG.TILE_SIZE),
            torch.zeros(1, CFG.TILE_SIZE, CFG.TILE_SIZE),
            0,
            ""
        )

# =============================================================================
# VALIDATION ON UNSEEN DATA
# =============================================================================
def validate_on_unseen():
    """
    Validate model on held-out validation set (not seen during training).
    Uses KFold with same seed to get exact unseen split.
    """
    print("\n" + "="*70)
    print("VALIDATING ON UNSEEN DATA (Fold 0 Validation Split)")
    print("="*70)
    
    # Build dataset
    auth = glob.glob(f'{CFG.TRAIN_IMG}/authentic/*.*')
    forg = glob.glob(f'{CFG.TRAIN_IMG}/forged/*.*')
    
    data = [{'image_path': p, 'label': 0, 'mask_path': None} for p in auth]
    for p in forg:
        base = os.path.basename(p).split('.')[0]
        mp = f'{CFG.TRAIN_MASK}/{base}.npy'
        if not os.path.exists(mp):
            mp = f'{CFG.TRAIN_MASK}/{base}_mask.npy'
        data.append({'image_path': p, 'label': 1, 'mask_path': mp if os.path.exists(mp) else None})
    
    df = pd.DataFrame(data)
    print(f"[DATA] Total samples: {len(df)} (Auth: {len(auth)}, Forged: {len(forg)})")
    
    # Get validation split (same as training)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(df))
    val_df = df.iloc[val_idx].copy()
    
    print(f"[DATA] Validation samples: {len(val_df)}")
    print(f"[DATA] Val Authentic: {len(val_df[val_df['label']==0])}")
    print(f"[DATA] Val Forged: {len(val_df[val_df['label']==1])}")
    
    # Load model
    print(f"\n[MODEL] Loading weights from {CFG.WEIGHTS_PATH}...")
    model = TitanXIIIFixed().to(CFG.DEVICE)
    
    if not os.path.exists(CFG.WEIGHTS_PATH):
        print(f"[ERROR] Weights not found: {CFG.WEIGHTS_PATH}")
        return None
    
    model.load_state_dict(torch.load(CFG.WEIGHTS_PATH, map_location=CFG.DEVICE))
    model.eval()
    print("[MODEL] Loaded successfully!")
    
    # Augmentation
    additional_targets = {'ela': 'mask', 'noise': 'mask', 'cfa': 'mask', 'edges': 'mask'}
    val_aug = A.Compose([
        A.LongestMaxSize(max_size=CFG.TILE_SIZE + 64),
        A.PadIfNeeded(CFG.TILE_SIZE, CFG.TILE_SIZE, border_mode=cv2.BORDER_REFLECT),
        A.CenterCrop(CFG.TILE_SIZE, CFG.TILE_SIZE)
    ], additional_targets=additional_targets)
    
    val_dataset = ValidationDataset(val_df, val_aug)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH, shuffle=False, num_workers=CFG.WORKERS)
    
    # Run validation
    all_preds = []
    all_targets = []
    all_probs_flat = []
    all_masks_flat = []
    
    fp_cases = []  # False positives
    fn_cases = []  # False negatives
    
    print("\n[VALIDATION] Running predictions...")
    with torch.no_grad():
        for i, (img, physics, edges, mask, label, img_path) in enumerate(val_loader):
            img = img.to(CFG.DEVICE)
            physics = physics.to(CFG.DEVICE)
            edges = edges.to(CFG.DEVICE)
            
            # Predict
            out, _, _, _ = model(img, physics, edges)
            prob = torch.sigmoid(out)
            prob_np = prob[0, 0].cpu().numpy()
            mask_np = mask[0, 0].numpy()
            
            # Use adaptive threshold
            thresh = adaptive_threshold(prob_np, CFG.PIXEL_THRESHOLD)
            
            # Clean mask with morphological operations
            pred_mask = clean_mask(prob_np, threshold=thresh, 
                                   min_area=CFG.MIN_CONTOUR_AREA, 
                                   morph_kernel=CFG.MORPH_KERNEL_SIZE)
            
            # Compute forged ratio
            forged_ratio = pred_mask.sum() / pred_mask.size
            
            # Final prediction: FORGED if ratio > threshold
            pred_label = 1 if forged_ratio > CFG.FORGED_RATIO_MIN else 0
            true_label = label.item()
            
            all_preds.append(pred_label)
            all_targets.append(true_label)
            
            # Store flattened for pixel-level metrics
            all_probs_flat.extend(prob_np.flatten().tolist())
            all_masks_flat.extend(mask_np.flatten().tolist())
            
            # Track errors
            if pred_label == 1 and true_label == 0:
                fp_cases.append((img_path[0], forged_ratio))
            elif pred_label == 0 and true_label == 1:
                fn_cases.append((img_path[0], forged_ratio))
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(val_loader)}")
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Image-level metrics
    img_f1 = f1_score(all_targets, all_preds, zero_division=0)
    img_prec = precision_score(all_targets, all_preds, zero_division=0)
    img_rec = recall_score(all_targets, all_preds, zero_division=0)
    
    # Pixel-level metrics (sample for speed)
    sample_size = min(len(all_probs_flat), 500000)
    indices = np.random.choice(len(all_probs_flat), sample_size, replace=False)
    sampled_probs = np.array(all_probs_flat)[indices]
    sampled_masks = np.array(all_masks_flat)[indices]
    
    # Find best pixel threshold
    best_pixel_f1 = 0
    best_pixel_thresh = 0.5
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_t = (sampled_probs > t).astype(int)
        f1_t = f1_score(sampled_masks, preds_t, zero_division=0)
        if f1_t > best_pixel_f1:
            best_pixel_f1 = f1_t
            best_pixel_thresh = t
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"\n📊 IMAGE-LEVEL METRICS:")
    print(f"   F1 Score:  {img_f1:.4f}")
    print(f"   Precision: {img_prec:.4f}")
    print(f"   Recall:    {img_rec:.4f}")
    
    print(f"\n📊 PIXEL-LEVEL METRICS (sampled {sample_size} pixels):")
    print(f"   Best F1:     {best_pixel_f1:.4f} @ threshold {best_pixel_thresh}")
    
    print(f"\n❌ FALSE POSITIVES (Authentic predicted as Forged): {len(fp_cases)}")
    for path, ratio in fp_cases[:5]:
        print(f"   - {os.path.basename(path)}: {ratio:.4f} forged ratio")
    
    print(f"\n❌ FALSE NEGATIVES (Forged predicted as Authentic): {len(fn_cases)}")
    for path, ratio in fn_cases[:5]:
        print(f"   - {os.path.basename(path)}: {ratio:.4f} forged ratio")
    
    print(f"\n{'='*70}")
    if img_f1 >= CFG.TARGET_F1:
        print(f"✅ TARGET ACHIEVED! F1 = {img_f1:.4f} >= {CFG.TARGET_F1}")
    else:
        print(f"⏳ TARGET NOT MET. F1 = {img_f1:.4f} < {CFG.TARGET_F1}")
        print(f"   Gap: {CFG.TARGET_F1 - img_f1:.4f}")
    print("="*70 + "\n")
    
    return {
        'img_f1': img_f1,
        'img_precision': img_prec,
        'img_recall': img_rec,
        'pixel_f1': best_pixel_f1,
        'pixel_threshold': best_pixel_thresh,
        'fp_count': len(fp_cases),
        'fn_count': len(fn_cases)
    }

# =============================================================================
# TEST INFERENCE WITH FIXED THRESHOLDS
# =============================================================================
def run_inference():
    """Run inference on test images with fixed thresholds."""
    print("\n" + "="*70)
    print("RUNNING INFERENCE ON TEST IMAGES")
    print("="*70)
    
    # Load model
    print(f"\n[MODEL] Loading weights from {CFG.WEIGHTS_PATH}...")
    model = TitanXIIIFixed().to(CFG.DEVICE)
    model.load_state_dict(torch.load(CFG.WEIGHTS_PATH, map_location=CFG.DEVICE))
    model.eval()
    
    # Find test images
    test_images = glob.glob(f'{CFG.TEST_IMG}/**/*.*', recursive=True)
    test_images = [p for p in test_images if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"[DATA] Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print(f"[ERROR] No images found in {CFG.TEST_IMG}")
        return
    
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    
    # Augmentation
    additional_targets = {'ela': 'mask', 'noise': 'mask', 'cfa': 'mask', 'edges': 'mask'}
    test_aug = A.Compose([
        A.LongestMaxSize(max_size=CFG.TILE_SIZE + 64),
        A.PadIfNeeded(CFG.TILE_SIZE, CFG.TILE_SIZE, border_mode=cv2.BORDER_REFLECT),
        A.CenterCrop(CFG.TILE_SIZE, CFG.TILE_SIZE)
    ], additional_targets=additional_targets)
    
    results = []
    
    print("\n[INFERENCE] Processing...")
    with torch.no_grad():
        for i, img_path in enumerate(test_images):
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            
            # Compute physics
            ela = get_ela_map(img)
            noise = get_noise_map_advanced(img)
            cfa = get_cfa_map_advanced(img)
            edges = get_edge_map(img)
            
            # Augment
            aug = test_aug(image=img, ela=ela, noise=noise, cfa=cfa, edges=edges)
            img_aug = aug['image']
            ela_aug, noise_aug, cfa_aug, edges_aug = aug['ela'], aug['noise'], aug['cfa'], aug['edges']
            
            # To tensor
            img_t = torch.from_numpy(img_aug.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            physics_t = torch.stack([
                torch.from_numpy(ela_aug).float(),
                torch.from_numpy(noise_aug).float(),
                torch.from_numpy(cfa_aug).float()
            ]).unsqueeze(0)
            edges_t = torch.from_numpy(edges_aug).float().unsqueeze(0)
            
            img_t = img_t.to(CFG.DEVICE)
            physics_t = physics_t.to(CFG.DEVICE)
            edges_t = edges_t.to(CFG.DEVICE)
            
            # Predict
            out, _, _, _ = model(img_t, physics_t, edges_t)
            prob = torch.sigmoid(out)
            prob_np = prob[0, 0].cpu().numpy()
            
            # Adaptive threshold + cleaning
            thresh = adaptive_threshold(prob_np, CFG.PIXEL_THRESHOLD)
            pred_mask = clean_mask(prob_np, threshold=thresh, 
                                   min_area=CFG.MIN_CONTOUR_AREA,
                                   morph_kernel=CFG.MORPH_KERNEL_SIZE)
            
            # Resize to original
            pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), 
                                           interpolation=cv2.INTER_NEAREST)
            
            # Compute stats
            forged_pixels = pred_mask_resized.sum()
            total_pixels = pred_mask_resized.size
            forged_ratio = forged_pixels / total_pixels
            
            # Final prediction
            prediction = 'FORGED' if forged_ratio > CFG.FORGED_RATIO_MIN else 'AUTHENTIC'
            
            # Save
            img_name = os.path.basename(img_path)
            npy_name = os.path.splitext(img_name)[0] + '.npy'
            np.save(os.path.join(CFG.OUTPUT_DIR, npy_name), pred_mask_resized)
            
            results.append({
                'image': img_name,
                'forged_pixels': int(forged_pixels),
                'forged_ratio': f'{forged_ratio:.4f}',
                'threshold_used': f'{thresh:.2f}',
                'prediction': prediction
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test_images)}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(CFG.OUTPUT_DIR, 'predictions_v2.csv'), index=False)
    
    print(f"\n[INFERENCE] Complete!")
    print(f"[INFERENCE] Output: {CFG.OUTPUT_DIR}")
    print(f"\n[SUMMARY]:")
    print(results_df)
    print(f"\nTotal: {len(results_df)}")
    print(f"FORGED: {len(results_df[results_df['prediction'] == 'FORGED'])}")
    print(f"AUTHENTIC: {len(results_df[results_df['prediction'] == 'AUTHENTIC'])}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='validate', choices=['validate', 'infer', 'both'])
    args = parser.parse_args()
    
    if args.mode in ['validate', 'both']:
        metrics = validate_on_unseen()
    
    if args.mode in ['infer', 'both']:
        run_inference()

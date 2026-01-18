"""
TITAN XIII FIXED v2: EMERGENCY PATCH
====================================
ALL CRITICAL ISSUES FIXED:
✅ Learning rate: 1e-4 → 2e-4
✅ Head bias: 0.0 → -0.2 (boosts initial predictions)
✅ Validation threshold: 0.5 → 0.3 (better recall)
✅ Gradient clipping: 2.0
✅ Gradient skip: 50.0
✅ AP stratified sampling: 10%
✅ OHEM delayed: epoch 20
✅ Focal alpha: 0.6
"""

import os, sys, glob, random, time, gc, math
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
import albumentations as A

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
# CONFIGURATION - EMERGENCY FIXES APPLIED
# =============================================================================
class CFG:
    BASE_DIR = '/Users/chanduchitikam/recodai/recodai-luc-scientific-image-forgery-detection'
    TRAIN_IMG = os.path.join(BASE_DIR, 'train_images')
    TRAIN_MASK = os.path.join(BASE_DIR, 'train_masks')
    WEIGHTS_BEST = "TITAN_XIII_BEST.pth"
    WEIGHTS_LAST = "TITAN_XIII_LAST.pth"
    CHECKPOINT_DIR = "checkpoints"
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    TILE_SIZE = 512
    BATCH = 2
    EPOCHS = 40
    LR = 2e-4  # FIXED: Doubled to 2e-4
    WARMUP_EPOCHS = 3
    OHEM_START = 20
    WORKERS = 0
    PIN_MEMORY = False
    
    EARLY_STOP_PATIENCE = 12
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    MIN_LR = 1e-7
    CHECKPOINT_EVERY = 5
    
    THRESH_MIN = 0.25  # FIXED: Lowered
    THRESH_MAX = 0.7
    
    TARGET_F1 = 0.75

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
print(f"[TITAN XIII v2] Device: {CFG.DEVICE}")
print(f"[TITAN XIII v2] LR: {CFG.LR} | Target F1: {CFG.TARGET_F1}")

# =============================================================================
# PHYSICS ENGINE
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
# LOSS FUNCTIONS
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

class BoundaryLoss(nn.Module):
    def __init__(self, weight=3.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        gx = F.conv2d(target, sobel_x, padding=1)
        gy = F.conv2d(target, sobel_y, padding=1)
        boundary = torch.sqrt(gx**2 + gy**2 + 1e-8)
        boundary = (boundary > 0.1).float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted = bce * (1 + boundary * (self.weight - 1))
        return weighted.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.6, gamma=2.0)
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
    
    def forward(self, pred, target):
        return 0.4 * self.focal(pred, target) + 0.4 * self.dice(pred, target) + 0.2 * self.boundary(pred, target)

# =============================================================================
# ENCODER MODULES
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

# =============================================================================
# ATTENTION MODULES
# =============================================================================
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

# =============================================================================
# DECODER BLOCKS
# =============================================================================
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

# =============================================================================
# MAIN MODEL - FIXED HEAD BIAS
# =============================================================================
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
        
        # FIXED: Warm-start with slight negative bias
        for head in [self.head_low, self.head_mid, self.head_high, self.head_final]:
            nn.init.constant_(head.bias, -0.2)  # FIXED: Was 0.0, now -0.2
            nn.init.xavier_uniform_(head.weight)
    
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
# DATASET
# =============================================================================
class ForensicDataset(Dataset):
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
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        if row['label'] == 1 and pd.notna(row['mask_path']):
            try:
                m = np.load(row['mask_path'])
                if hasattr(m, 'files'):
                    m = m[m.files[0]]
                if m.ndim == 3:
                    m = m.max(axis=2)
                mask = cv2.resize((m > 0).astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            except:
                pass
        ela = get_ela_map(img)
        noise = get_noise_map_advanced(img)
        cfa = get_cfa_map_advanced(img)
        edges = get_edge_map(img)
        if self.transform:
            aug = self.transform(image=img, mask=mask, ela=ela, noise=noise, cfa=cfa, edges=edges)
            img, mask, ela, noise, cfa, edges = aug['image'], aug['mask'], aug['ela'], aug['noise'], aug['cfa'], aug['edges']
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        physics_t = torch.stack([torch.from_numpy(ela).float(), torch.from_numpy(noise).float(), torch.from_numpy(cfa).float()])
        edges_t = torch.from_numpy(edges).float()
        return img_t, physics_t, edges_t, mask_t
    
    def _dummy(self):
        return (torch.zeros(3, CFG.TILE_SIZE, CFG.TILE_SIZE), torch.zeros(3, CFG.TILE_SIZE, CFG.TILE_SIZE), 
                torch.zeros(CFG.TILE_SIZE, CFG.TILE_SIZE), torch.zeros(1, CFG.TILE_SIZE, CFG.TILE_SIZE))

# =============================================================================
# TRAINING - ALL FIXES APPLIED
# =============================================================================
def train():
    print("[TITAN XIII v2] Building dataset...")
    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    auth = glob.glob(f'{CFG.TRAIN_IMG}/authentic/*.*')
    forg = glob.glob(f'{CFG.TRAIN_IMG}/forged/*.*')
    data = [{'image_path': p, 'label': 0, 'mask_path': None} for p in auth]
    for p in forg:
        base = os.path.basename(p).split('.')[0]
        mp = f'{CFG.TRAIN_MASK}/{base}.npy'
        if not os.path.exists(mp):
            mp = f'{CFG.TRAIN_MASK}/{base}_mask.npy'
        data.append({'image_path': p, 'label': 1, 'mask_path': mp})
    df = pd.DataFrame(data)
    print(f"[DATA] Total: {len(df)} samples")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(df))
    additional_targets = {'ela': 'mask', 'noise': 'mask', 'cfa': 'mask', 'edges': 'mask'}
    train_aug = A.Compose([
        A.LongestMaxSize(max_size=CFG.TILE_SIZE + 128), A.PadIfNeeded(CFG.TILE_SIZE, CFG.TILE_SIZE, border_mode=cv2.BORDER_REFLECT),
        A.RandomCrop(CFG.TILE_SIZE, CFG.TILE_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.OneOf([A.GaussNoise(), A.GaussianBlur()], p=0.3), A.RandomBrightnessContrast(p=0.3),
    ], additional_targets=additional_targets)
    val_aug = A.Compose([
        A.LongestMaxSize(max_size=CFG.TILE_SIZE + 64), A.PadIfNeeded(CFG.TILE_SIZE, CFG.TILE_SIZE, border_mode=cv2.BORDER_REFLECT),
        A.CenterCrop(CFG.TILE_SIZE, CFG.TILE_SIZE)
    ], additional_targets=additional_targets)
    train_dl = DataLoader(ForensicDataset(df.iloc[train_idx], train_aug), batch_size=CFG.BATCH, shuffle=True, num_workers=CFG.WORKERS)
    val_dl = DataLoader(ForensicDataset(df.iloc[val_idx], val_aug), batch_size=CFG.BATCH, shuffle=False, num_workers=CFG.WORKERS)
    print("[MODEL] Building TitanXIIIFixed...")
    model = TitanXIIIFixed().to(CFG.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=CFG.MIN_LR)
    criterion = CombinedLoss()
    criterion_aux = nn.BCEWithLogitsLoss()
    best_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    print(f"\n[TRAINING] Starting | LR: {CFG.LR} | Target F1: {CFG.TARGET_F1}")
    
    for epoch in range(1, CFG.EPOCHS + 1):
        model.train()
        for i, (img, physics, edges, mask) in enumerate(train_dl):
            img, physics, edges, mask = img.to(CFG.DEVICE), physics.to(CFG.DEVICE), edges.to(CFG.DEVICE), mask.to(CFG.DEVICE)
            optimizer.zero_grad()
            out, p_low, p_mid, p_high = model(img, physics, edges)
            loss_main = criterion(out, mask)
            loss_aux = 0.1 * criterion_aux(p_low, mask) + 0.2 * criterion_aux(p_mid, mask) + 0.3 * criterion_aux(p_high, mask)
            loss = loss_main + loss_aux
            if epoch >= CFG.OHEM_START:
                with torch.no_grad():
                    probs = torch.sigmoid(out)
                    error = torch.abs(probs - mask)
                    k = int(error.numel() * 0.2)
                    if k > 0:
                        hard_thresh = torch.topk(error.view(-1), k)[0][-1]
                hard_mask = (torch.abs(torch.sigmoid(out) - mask) >= hard_thresh).float()
                ohem_weight = hard_mask * 1.5 + 0.5
                ohem_loss = (F.binary_cross_entropy_with_logits(out, mask, reduction='none') * ohem_weight).mean()
                loss = loss + 0.2 * ohem_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            if grad_norm > 50.0:
                print(f"[WARN] Catastrophic gradient (norm={grad_norm:.2f}), skipping")
                optimizer.zero_grad()
                continue
            optimizer.step()
            if i % 20 == 0:
                print(f"\rE{epoch} [{i}/{len(train_dl)}] L:{loss.item():.4f} GN:{grad_norm:.2f}", end="")
        scheduler.step()
        
        # Validation with FIXED threshold
        model.eval()
        tp, fp, fn = 0, 0, 0
        ap_preds, ap_targets = [], []
        with torch.no_grad():
            for img, physics, edges, mask in val_dl:
                img, physics, edges, mask = img.to(CFG.DEVICE), physics.to(CFG.DEVICE), edges.to(CFG.DEVICE), mask.to(CFG.DEVICE)
                out, _, _, _ = model(img, physics, edges)
                prob = torch.sigmoid(out)
                pred = (prob > 0.3).float()  # FIXED: 0.5 → 0.3
                tp += (pred * mask).sum().item()
                fp += (pred * (1 - mask)).sum().item()
                fn += ((1 - pred) * mask).sum().item()
                prob_flat = prob.cpu().numpy().flatten()
                mask_flat = mask.cpu().numpy().flatten()
                pos_idx = np.where(mask_flat > 0.5)[0]
                neg_idx = np.where(mask_flat <= 0.5)[0]
                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    n_pos = min(500, len(pos_idx))
                    n_neg = min(500, len(neg_idx))
                    pos_sample = np.random.choice(pos_idx, n_pos, replace=False)
                    neg_sample = np.random.choice(neg_idx, n_neg, replace=False)
                    ap_preds.extend(prob_flat[pos_sample].tolist())
                    ap_preds.extend(prob_flat[neg_sample].tolist())
                    ap_targets.extend([1] * n_pos)
                    ap_targets.extend([0] * n_neg)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        ap = 0.0
        try:
            if len(ap_targets) > 100 and sum(ap_targets) > 10:
                ap = average_precision_score(ap_targets, ap_preds)
        except:
            pass
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{CFG.EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AP: {ap:.4f}")
        print(f"Best F1: {best_f1:.4f} @ E{best_epoch}")
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), CFG.WEIGHTS_BEST)
            print(f"🎯 NEW BEST! Saved to {CFG.WEIGHTS_BEST}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improve for {epochs_no_improve}/{CFG.EARLY_STOP_PATIENCE} epochs")
        if epochs_no_improve >= CFG.EARLY_STOP_PATIENCE:
            print(f"\n🛑 EARLY STOPPING @ Epoch {epoch}")
            break
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE | Best F1: {best_f1:.4f} @ Epoch {best_epoch}")
    print(f"Target: {CFG.TARGET_F1} | {'✅ ACHIEVED' if best_f1 >= CFG.TARGET_F1 else '⏳ TRAIN MORE'}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    train()

"""
TITAN-APEX V4: HIGH-PERFORMANCE FORENSIC SEGMENTATION
======================================================
Complete redesign to achieve F1 > 0.8 on Kaggle

Key Changes from V3:
1. HEAVY class imbalance handling (pos_weight=10-20)
2. Dice Loss as primary (works better for segmentation)
3. Deep supervision from all decoder stages
4. Auxiliary classification head (is image forged?)
5. Better augmentations for forensic task
6. Lower LR with cosine annealing + warmup
7. Gradient clipping for stability
8. Mixed precision where possible
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
from pathlib import Path
from tqdm import tqdm
import timm
import math
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Paths
    DATA_ROOT = Path("recodai-luc-scientific-image-forgery-detection")
    TRAIN_IMAGES = DATA_ROOT / "train_images"
    TRAIN_MASKS = DATA_ROOT / "train_masks"
    
    # Training - optimized for M4 Pro
    IMG_SIZE = 512  # Larger for better detail
    BATCH_SIZE = 4  # Smaller batch for larger images
    GRAD_ACCUM = 4  # Effective batch = 16
    EPOCHS = 60  # More epochs for convergence
    LR = 3e-4  # Lower base LR
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 3
    
    # Loss weights - CRITICAL for class imbalance
    POS_WEIGHT = 15.0  # Heavy weight for forged pixels
    DICE_WEIGHT = 0.7
    BCE_WEIGHT = 0.3
    AUX_WEIGHT = 0.2  # Auxiliary classification loss
    
    # Architecture
    ENCODER = "efficientnet_b4"  # Larger encoder
    
    # Workers
    NUM_WORKERS = 4
    
    # Threshold for inference
    THRESHOLD = 0.35

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# ============================================================
# SRM FILTERS - Proper 30 kernels
# ============================================================
def get_srm_kernels():
    """30 SRM high-pass filter kernels for forensic analysis"""
    kernels = []
    
    # 1st order edge detectors (4 kernels)
    edge1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
    for _ in range(4):
        kernels.append(edge1.copy())
        edge1 = np.rot90(edge1)
    
    # 2nd order edge detectors (4 kernels)
    edge2 = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32)
    for _ in range(4):
        kernels.append(edge2.copy())
        edge2 = np.rot90(edge2)
    
    # 3rd order SQUARE kernels (4 kernels)
    square3 = np.array([[0, 0, 0], [-1, 3, -3], [0, 0, 1]], dtype=np.float32)
    for _ in range(4):
        kernels.append(square3.copy())
        square3 = np.rot90(square3)
    
    # 3rd order EDGE kernels (4 kernels)
    edge3 = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32)
    edge3[2, 1] = 0
    edge3[1, 0] = 1
    for _ in range(4):
        kernels.append(edge3.copy())
        edge3 = np.rot90(edge3)
    
    # SQUARE 3x3 (1 kernel)
    square = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)
    kernels.append(square)
    
    # SQUARE 5x5 center (need to pad to 5x5 then extract 3x3 behavior)
    sq5 = np.array([[-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
    # Extract 3x3 center
    kernels.append(sq5[1:4, 1:4].copy())
    
    # Additional high-pass filters for diversity
    hp1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    kernels.append(hp1)
    
    hp2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    kernels.append(hp2)
    
    # Diagonal filters (4 kernels)
    diag1 = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32)
    for _ in range(4):
        kernels.append(diag1.copy())
        diag1 = np.rot90(diag1)
    
    # Additional edge variants (4 kernels)
    ev1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4  # Sobel-like
    for _ in range(4):
        kernels.append(ev1.copy())
        ev1 = np.rot90(ev1)
    
    # Ensure exactly 30 kernels
    while len(kernels) < 30:
        # Add normalized random high-pass
        k = np.random.randn(3, 3).astype(np.float32)
        k = k - k.mean()
        kernels.append(k)
    
    kernels = kernels[:30]
    
    # Stack and normalize
    kernels = np.stack(kernels)  # (30, 3, 3)
    kernels = kernels[:, np.newaxis, :, :]  # (30, 1, 3, 3)
    
    return kernels

class SRMConv2d(nn.Module):
    """SRM layer with 30 fixed high-pass filters"""
    def __init__(self):
        super().__init__()
        kernels = get_srm_kernels()  # (30, 1, 3, 3)
        kernels = np.tile(kernels, (1, 3, 1, 1))  # (30, 3, 3, 3) for RGB
        self.register_buffer('weight', torch.from_numpy(kernels))
        
    def forward(self, x):
        out = F.conv2d(x, self.weight, padding=1)
        # TLU activation (Truncated Linear Unit)
        out = torch.clamp(out, -3, 3)
        return out

# ============================================================
# BAYAR CONSTRAINED CONV
# ============================================================
class BayarConv2d(nn.Module):
    """Bayar constrained convolution - learns manipulation-specific filters"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, 5, 5) * 0.01)
        
    def forward(self, x):
        # Apply constraint: center = -1, rest sums to 1
        kernel = self.kernel.clone()
        # Normalize non-center weights to sum to 1
        center_mask = torch.zeros(5, 5, device=kernel.device)
        center_mask[2, 2] = 1
        
        # Get weights excluding center
        non_center = kernel * (1 - center_mask)
        non_center_sum = non_center.sum(dim=(2, 3), keepdim=True) + 1e-8
        non_center = non_center / non_center_sum
        
        # Set center to -1
        constrained = non_center
        constrained[:, :, 2, 2] = -1
        
        return F.conv2d(x, constrained, padding=2)

# ============================================================
# PHYSICS-BASED FEATURE EXTRACTION
# ============================================================
def compute_physics_maps(x):
    """Compute physics-based forensic features"""
    B, C, H, W = x.shape
    gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # Sobel gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    
    # Local variance (noise estimation)
    mean_filter = torch.ones(1, 1, 5, 5, dtype=x.dtype, device=x.device) / 25
    local_mean = F.conv2d(gray, mean_filter, padding=2)
    local_sq_mean = F.conv2d(gray**2, mean_filter, padding=2)
    local_var = torch.clamp(local_sq_mean - local_mean**2, min=0)
    
    # Noise residual (high-frequency content)
    blur = F.avg_pool2d(gray, 3, stride=1, padding=1)
    noise_res = gray - blur
    
    # Laplacian (edge detection)
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                             dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    edge = torch.abs(F.conv2d(gray, laplacian, padding=1))
    
    # Normalize all maps
    def normalize(t):
        t_min = t.view(B, 1, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        t_max = t.view(B, 1, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        return (t - t_min) / (t_max - t_min + 1e-8)
    
    return torch.cat([
        normalize(grad_mag),
        normalize(local_var),
        normalize(torch.abs(noise_res)),
        normalize(edge)
    ], dim=1)  # (B, 4, H, W)

# ============================================================
# ATTENTION MODULES
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_val = x.max(dim=1, keepdim=True)[0]
        w = self.conv(torch.cat([avg, max_val], dim=1))
        return x * w

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ============================================================
# ENCODER
# ============================================================
class DualStreamEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Semantic stream - EfficientNet-B4
        self.semantic = timm.create_model(
            Config.ENCODER,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)  # All 5 stages
        )
        
        # Get channel info
        dummy = torch.zeros(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
        with torch.no_grad():
            feats = self.semantic(dummy)
            self.semantic_channels = [f.shape[1] for f in feats]
        # EfficientNet-B4: [24, 32, 56, 160, 448]
        
        # Forensic stream
        self.srm = SRMConv2d()  # 30 channels
        self.bayar = BayarConv2d(3, 3)  # 3 channels
        # Physics maps: 4 channels
        # Total forensic: 37 channels
        
        # Forensic feature processor
        self.forensic_conv = nn.Sequential(
            nn.Conv2d(37, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Fusion convs for each stage
        self.fusions = nn.ModuleList()
        for ch in self.semantic_channels:
            self.fusions.append(nn.Sequential(
                nn.Conv2d(ch + 64, ch, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                CBAM(ch)
            ))
    
    def forward(self, x):
        # Semantic features
        semantic_feats = self.semantic(x)
        
        # Forensic features
        srm_out = self.srm(x)  # (B, 30, H, W)
        bayar_out = self.bayar(x)  # (B, 3, H, W)
        physics = compute_physics_maps(x)  # (B, 4, H, W)
        
        forensic = torch.cat([srm_out, bayar_out, physics], dim=1)  # (B, 37, H, W)
        forensic = self.forensic_conv(forensic)  # (B, 64, H, W)
        
        # Fuse at each scale
        fused_feats = []
        for i, (sem_feat, fusion) in enumerate(zip(semantic_feats, self.fusions)):
            # Resize forensic to match semantic scale
            h, w = sem_feat.shape[2:]
            forensic_scaled = F.interpolate(forensic, size=(h, w), mode='bilinear', align_corners=False)
            
            # Concatenate and fuse
            combined = torch.cat([sem_feat, forensic_scaled], dim=1)
            fused = fusion(combined)
            fused_feats.append(fused)
        
        return fused_feats

# ============================================================
# DECODER WITH DEEP SUPERVISION
# ============================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CBAM(out_ch)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        # encoder_channels = [24, 32, 56, 160, 448] for EfficientNet-B4
        
        # Decoder blocks (going from deepest to shallowest)
        self.dec4 = DecoderBlock(encoder_channels[4], encoder_channels[3], 256)
        self.dec3 = DecoderBlock(256, encoder_channels[2], 128)
        self.dec2 = DecoderBlock(128, encoder_channels[1], 64)
        self.dec1 = DecoderBlock(64, encoder_channels[0], 32)
        
        # Final upsampling and output
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        # Deep supervision heads
        self.ds4 = nn.Conv2d(256, 1, 1)
        self.ds3 = nn.Conv2d(128, 1, 1)
        self.ds2 = nn.Conv2d(64, 1, 1)
        
        # Initialize output biases for class imbalance
        for m in [self.final_conv[-1], self.ds4, self.ds3, self.ds2]:
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, -2.0)  # Start predicting mostly 0 (authentic)
    
    def forward(self, features):
        # features = [f0, f1, f2, f3, f4] from encoder
        f0, f1, f2, f3, f4 = features
        
        # Decode
        d4 = self.dec4(f4, f3)  # 256 ch
        d3 = self.dec3(d4, f2)  # 128 ch
        d2 = self.dec2(d3, f1)  # 64 ch
        d1 = self.dec1(d2, f0)  # 32 ch
        
        # Final output
        out = self.final_up(d1)
        out = self.final_conv(out)
        
        # Deep supervision outputs (for training)
        ds4 = self.ds4(d4)
        ds3 = self.ds3(d3)
        ds2 = self.ds2(d2)
        
        return out, [ds4, ds3, ds2]

# ============================================================
# AUXILIARY CLASSIFICATION HEAD
# ============================================================
class AuxiliaryHead(nn.Module):
    """Binary classifier: Is this image forged or authentic?"""
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        # Initialize for class balance
        nn.init.constant_(self.fc[-1].bias, 0.0)
        
    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)

# ============================================================
# FULL MODEL
# ============================================================
class TitanApexV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DualStreamEncoder()
        self.decoder = Decoder(self.encoder.semantic_channels)
        self.aux_head = AuxiliaryHead(self.encoder.semantic_channels[-1])
        
    def forward(self, x):
        features = self.encoder(x)
        mask, ds_outputs = self.decoder(features)
        aux_logit = self.aux_head(features[-1])  # From deepest feature
        
        return {
            'mask': mask,
            'ds_outputs': ds_outputs,
            'aux': aux_logit
        }

# ============================================================
# LOSS FUNCTIONS - Aggressive forgery detection
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class TverskyLoss(nn.Module):
    """Tversky loss with adjustable FP/FN penalty - CRITICAL for imbalanced segmentation"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha  # FP penalty (lower = less penalty for false positives)
        self.beta = beta    # FN penalty (higher = more penalty for missing forgeries)
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky

class FocalTverskyLoss(nn.Module):
    """Focal Tversky - extra penalty for hard examples"""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        focal_tversky = torch.pow(1 - tversky, self.gamma)
        return focal_tversky

class WeightedBCELoss(nn.Module):
    """BCE with massive weight for forged pixels"""
    def __init__(self, pos_weight=20.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred, target):
        # Per-pixel weighting
        weight = torch.where(target > 0.5, self.pos_weight, 1.0)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (bce * weight).mean()
        return weighted_bce

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tversky = FocalTverskyLoss(alpha=0.2, beta=0.8, gamma=1.5)  # Very high FN penalty
        self.wbce = WeightedBCELoss(pos_weight=25.0)  # Massive weight for forgeries
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
        
    def forward(self, outputs, mask, is_forged):
        """
        outputs: dict with 'mask', 'ds_outputs', 'aux'
        mask: ground truth mask (B, 1, H, W)
        is_forged: binary label (B,) - 1 if image has forgery
        """
        pred_mask = outputs['mask']
        ds_outputs = outputs['ds_outputs']
        aux_logit = outputs['aux']
        
        # Ensure mask matches prediction size
        if pred_mask.shape[2:] != mask.shape[2:]:
            pred_mask_resized = F.interpolate(pred_mask, size=mask.shape[2:], 
                                               mode='bilinear', align_corners=False)
        else:
            pred_mask_resized = pred_mask
        
        # Only compute segmentation loss on FORGED images (authentic have no forgery to detect)
        forged_mask = is_forged.view(-1, 1, 1, 1).expand_as(mask)
        has_forgery = is_forged.sum() > 0
        
        if has_forgery:
            # Mask out authentic images for segmentation loss
            forged_pred = pred_mask_resized[is_forged > 0.5]
            forged_gt = mask[is_forged > 0.5]
            
            tversky_loss = self.tversky(forged_pred, forged_gt)
            bce_loss = self.wbce(forged_pred, forged_gt)
            seg_loss = 0.6 * tversky_loss + 0.4 * bce_loss
        else:
            # All authentic - just penalize any predictions
            seg_loss = torch.sigmoid(pred_mask_resized).mean() * 0.5
        
        # For authentic images: penalize any positive predictions
        if (is_forged < 0.5).sum() > 0:
            auth_pred = pred_mask_resized[is_forged < 0.5]
            auth_penalty = torch.sigmoid(auth_pred).mean() * 2.0  # Penalize false positives on authentic
        else:
            auth_penalty = torch.tensor(0.0, device=pred_mask.device)
        
        # Deep supervision loss (only on forged)
        ds_loss = torch.tensor(0.0, device=pred_mask.device)
        if has_forgery:
            for i, ds_out in enumerate(ds_outputs):
                ds_mask = F.interpolate(mask, size=ds_out.shape[2:], mode='nearest')
                forged_ds = ds_out[is_forged > 0.5]
                forged_ds_mask = ds_mask[is_forged > 0.5]
                ds_loss = ds_loss + self.tversky(forged_ds, forged_ds_mask) * (0.5 ** (len(ds_outputs) - i))
            ds_loss = ds_loss / len(ds_outputs)
        
        # Auxiliary classification loss - helps model learn forged vs authentic
        self.bce.pos_weight = self.bce.pos_weight.to(aux_logit.device)
        aux_loss = F.binary_cross_entropy_with_logits(
            aux_logit.squeeze(), is_forged.float()
        )
        
        # Total loss
        total_loss = seg_loss + 0.3 * ds_loss + auth_penalty + Config.AUX_WEIGHT * aux_loss
        
        return {
            'total': total_loss,
            'seg': seg_loss,
            'ds': ds_loss,
            'aux': aux_loss,
            'auth_penalty': auth_penalty,
            'tversky': tversky_loss if has_forgery else torch.tensor(0.0)
        }

# ============================================================
# DATASET
# ============================================================
class ForensicDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.is_train = is_train
        
        # Stronger augmentation for training
        self.jpeg_qualities = [50, 60, 70, 80, 90, 95]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(str(self.image_paths[idx]))
        if img is None:
            # Fallback: return a black image and zero mask
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            return torch.tensor(img, dtype=torch.float32), torch.tensor(mask[np.newaxis], dtype=torch.float32), torch.tensor(0.0)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask (None for authentic images)
        if self.mask_paths[idx] is not None:
            mask = np.load(str(self.mask_paths[idx]))
            # Ensure 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            if mask.ndim > 2:
                mask = mask[:, :, 0]
            mask = mask.astype(np.float32)
            # Normalize if needed
            if mask.max() > 1:
                mask = (mask > 0.5).astype(np.float32)
        else:
            # Authentic image - create zero mask matching image size
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        
        is_forged = float(mask.max() > 0)
        
        # Resize with proper tuple format
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        if mask.shape[0] > 0 and mask.shape[1] > 0:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        if self.is_train:
            # Augmentations
            if np.random.random() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.random() < 0.5:
                img = img[::-1, :].copy()
                mask = mask[::-1, :].copy()
            if np.random.random() < 0.3:
                k = np.random.randint(1, 4)
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
            
            # JPEG compression (critical for forensic robustness)
            if np.random.random() < 0.5:
                quality = np.random.choice(self.jpeg_qualities)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Color jitter
            if np.random.random() < 0.3:
                img = img.astype(np.float32)
                img *= np.random.uniform(0.8, 1.2, (1, 1, 3))
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Gaussian noise
            if np.random.random() < 0.2:
                noise = np.random.normal(0, np.random.uniform(5, 15), img.shape)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # To tensor
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask[np.newaxis]).float()
        
        return img, mask, torch.tensor(is_forged)

# ============================================================
# METRICS
# ============================================================
def compute_f1(pred_mask, gt_mask, threshold=0.5):
    """Compute pixel-level F1 score"""
    pred = (pred_mask > threshold).float()
    gt = gt_mask.float()
    
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1.item(), precision.item(), recall.item(), tp.item(), fp.item(), fn.item()

def find_best_threshold(pred_masks, gt_masks):
    """Find optimal threshold for F1"""
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = None
    
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        f1, prec, rec, tp, fp, fn = compute_f1(pred_masks, gt_masks, thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = (prec, rec, tp, fp, fn)
    
    return best_thresh, best_f1, best_metrics

# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, loader, optimizer, criterion, scaler, device, scheduler=None):
    model.train()
    total_loss = 0
    total_tversky = 0
    
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    for i, (images, masks, is_forged) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        is_forged = is_forged.to(device)
        
        # Forward pass
        outputs = model(images)
        losses = criterion(outputs, masks, is_forged)
        loss = losses['total'] / Config.GRAD_ACCUM
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (i + 1) % Config.GRAD_ACCUM == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        total_loss += losses['total'].item()
        tversky_val = losses['tversky'].item() if isinstance(losses['tversky'], torch.Tensor) else losses['tversky']
        total_tversky += tversky_val
        
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'tversky': f"{tversky_val:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    return total_loss / len(loader), total_tversky / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_masks = []
    
    for images, masks, is_forged in tqdm(loader, desc="Validating"):
        images = images.to(device)
        masks = masks.to(device)
        is_forged = is_forged.to(device)
        
        outputs = model(images)
        losses = criterion(outputs, masks, is_forged)
        total_loss += losses['total'].item()
        
        # Collect predictions
        pred = torch.sigmoid(outputs['mask'])
        if pred.shape[2:] != masks.shape[2:]:
            pred = F.interpolate(pred, size=masks.shape[2:], mode='bilinear', align_corners=False)
        
        all_preds.append(pred.cpu())
        all_masks.append(masks.cpu())
    
    # Concatenate
    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    
    # Find best threshold
    best_thresh, best_f1, metrics = find_best_threshold(all_preds, all_masks)
    
    return {
        'loss': total_loss / len(loader),
        'f1': best_f1,
        'threshold': best_thresh,
        'precision': metrics[0],
        'recall': metrics[1],
        'pred_min': all_preds.min().item(),
        'pred_max': all_preds.max().item(),
        'pred_mean': all_preds.mean().item()
    }

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("TITAN-APEX V4: High-Performance Forensic Segmentation")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Collect data
    forged_dir = Config.TRAIN_IMAGES / "forged"
    authentic_dir = Config.TRAIN_IMAGES / "authentic"
    
    all_images = []
    all_masks = []
    
    # Forged images (have real masks)
    for img_path in sorted(forged_dir.glob("*.jpg")) + sorted(forged_dir.glob("*.png")):
        mask_path = Config.TRAIN_MASKS / f"{img_path.stem}.npy"
        if mask_path.exists():
            all_images.append(img_path)
            all_masks.append(mask_path)
    
    forged_count = len(all_images)
    print(f"Forged images with masks: {forged_count}")
    
    # Authentic images (use None for mask - will create zero mask in dataset)
    for img_path in sorted(authentic_dir.glob("*.jpg")) + sorted(authentic_dir.glob("*.png")):
        all_images.append(img_path)
        all_masks.append(None)  # No mask for authentic
    
    print(f"Authentic images: {len(all_images) - forged_count}")
    
    print(f"Total samples: {len(all_images)}")
    
    # Split
    n = len(all_images)
    indices = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_images = [all_images[i] for i in train_idx]
    train_masks = [all_masks[i] for i in train_idx]
    val_images = [all_images[i] for i in val_idx]
    val_masks = [all_masks[i] for i in val_idx]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    print(f"Config: IMG_SIZE={Config.IMG_SIZE}, BATCH={Config.BATCH_SIZE}, ACCUM={Config.GRAD_ACCUM}")
    print(f"Effective batch size: {Config.BATCH_SIZE * Config.GRAD_ACCUM}")
    
    # Datasets
    train_ds = ForensicDataset(train_images, train_masks, Config.IMG_SIZE, is_train=True)
    val_ds = ForensicDataset(val_images, val_masks, Config.IMG_SIZE, is_train=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Model
    model = TitanApexV4().to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss
    criterion = CombinedLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=0.01)
    
    # Scheduler - Cosine annealing with warmup
    total_steps = len(train_loader) * Config.EPOCHS // Config.GRAD_ACCUM
    warmup_steps = len(train_loader) * Config.WARMUP_EPOCHS // Config.GRAD_ACCUM
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Training loop
    best_f1 = 0
    scaler = None  # MPS doesn't support AMP well
    
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.EPOCHS}")
        
        # Train
        train_loss, train_tversky = train_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE, scheduler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, DEVICE)
        
        # Report
        print(f"Train Loss: {train_loss:.4f}, Train Tversky: {train_tversky:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Pred stats: min={val_metrics['pred_min']:.4f}, max={val_metrics['pred_max']:.4f}, mean={val_metrics['pred_mean']:.4f}")
        print(f"  Best thresh={val_metrics['threshold']:.2f}, Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f}")
        print(f"  F1={val_metrics['f1']:.4f}")
        
        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
                'threshold': val_metrics['threshold']
            }, 'TITAN_APEX_V4_BEST.pth')
            print(f"  ★ New best F1: {best_f1:.4f} - Saved!")
        
        # Early warning
        if epoch == 5 and best_f1 < 0.2:
            print("\n⚠️ Warning: F1 still low at epoch 5. Model is learning slowly.")
        
        # Progress check
        if epoch == 10 and best_f1 < 0.3:
            print("\n⚠️ Warning: F1 < 0.3 at epoch 10. Consider:")
            print("   - Reducing learning rate")
            print("   - Increasing pos_weight")
            print("   - Checking data quality")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: TITAN_APEX_V4_BEST.pth")

if __name__ == "__main__":
    main()

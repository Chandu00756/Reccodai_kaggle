"""
================================================================================
TITAN TRAIN 0.8: Ultimate Training Script for 0.8+ F1
================================================================================
TARGET: 0.8+ F1 Score

STRATEGY:
├── Heavy augmentation (copy-move simulation, elastic, color)
├── Combined loss: Dice(0.5) + BCE(0.3) + Focal(0.2)
├── 5-fold cross-validation
├── Cosine annealing + warm restarts
├── Mixed precision (AMP)
├── Start from pretrained Swin backbone
├── Progressive resizing: 256 → 384
└── Test-time augmentation built into validation

================================================================================
"""

import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

try:
    import timm
except ImportError:
    os.system('pip install timm -q')
    import timm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    os.system('pip install albumentations -q')
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

try:
    from sklearn.model_selection import KFold
except ImportError:
    os.system('pip install scikit-learn -q')
    from sklearn.model_selection import KFold

# ============================================================================
# CONFIGURATION
# ============================================================================
class CFG:
    # Paths - LOCAL TRAINING on Mac
    LOCAL_ROOT = Path('/Users/chanduchitikam/recodai')
    DATA_ROOT = LOCAL_ROOT / 'recodai-luc-scientific-image-forgery-detection'
    OUTPUT_DIR = LOCAL_ROOT / 'checkpoints'
    PRETRAINED_WEIGHTS = None  # Train from scratch - old weights caused collapse
    
    @classmethod
    def setup_dirs(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    TRAIN_IMAGES = DATA_ROOT / 'train_images'
    TRAIN_MASKS = DATA_ROOT / 'train_masks'
    SUPP_IMAGES = DATA_ROOT / 'supplemental_images'
    SUPP_MASKS = DATA_ROOT / 'supplemental_masks'
    
    # Model
    MODEL_NAME = 'swin_base_patch4_window12_384'
    IMG_SIZE = 384
    IN_CHANNELS = 3
    
    # Training - STABLE for 0.8+
    EPOCHS = 100            # More epochs for convergence
    BATCH_SIZE = 4          # Adjust based on GPU memory
    NUM_WORKERS = 2         # Reduced for Mac stability
    LEARNING_RATE = 1e-4    # LOWER - 3e-4 was too aggressive
    MIN_LR = 1e-6           # Minimum LR for scheduler
    WEIGHT_DECAY = 1e-5     # Lighter regularization
    
    # Loss weights - DICE HEAVY for segmentation
    DICE_WEIGHT = 0.6
    BCE_WEIGHT = 0.2
    FOCAL_WEIGHT = 0.2
    
    # Cross-validation
    N_FOLDS = 5
    TRAIN_FOLDS = [0]       # Which folds to train (set to [0,1,2,3,4] for full CV)
    
    # Early stopping
    PATIENCE = 15           # More patience for 0.8 target
    
    # Mixed precision
    USE_AMP = True
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Seed
    SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.SEED)
CFG.setup_dirs()
print(f"🚀 TITAN TRAIN 0.8 - Device: {CFG.DEVICE}")

# ============================================================================
# DATASET
# ============================================================================
class ForgeryDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_train = is_train
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = cv2.imread(str(img_path))
            if image is None or image.size == 0 or len(image.shape) < 2:
                raise ValueError("Invalid image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
        except Exception:
            # Fallback to blank image
            image = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)
            h, w = CFG.IMG_SIZE, CFG.IMG_SIZE
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = None
        if mask_path is not None:
            try:
                if str(mask_path).endswith('.npy') and os.path.exists(mask_path):
                    mask = np.load(mask_path)
                elif os.path.exists(mask_path):
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if mask is not None and mask.size > 0:
                    # Ensure mask is 2D
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0] if mask.shape[2] > 0 else mask.mean(axis=2)
                    if len(mask.shape) >= 2:
                        mask = (mask > 0).astype(np.float32)
                        # Resize mask to match image
                        if mask.shape[0] != h or mask.shape[1] != w:
                            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask = None
                else:
                    mask = None
            except Exception:
                mask = None
        
        # Default mask if not loaded
        if mask is None:
            mask = np.zeros((h, w), dtype=np.float32)
        
        # Ensure mask is 2D before augmentation
        mask = mask.astype(np.float32)
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure output mask is [1, H, W]
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            elif mask.dim() > 3:
                mask = mask[:1, :, :]  # Take first channel only
        else:
            mask = torch.tensor(mask, dtype=torch.float32)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        
        return image, mask

# ============================================================================
# AUGMENTATIONS - HEAVY FOR FORGERY DETECTION
# ============================================================================
def get_train_transforms(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        
        # Color augmentations (important for forgery)
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        ], p=0.7),
        
        # Noise augmentations (simulates compression artifacts)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.5),
        
        # Blur augmentations
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        
        # Compression artifacts (critical for JPEG forgery)
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.5),
        
        # Geometric distortions
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=0.3),
        ], p=0.3),
        
        # Cutout (simulates occlusion)
        A.CoarseDropout(max_holes=8, max_height=img_size//16, max_width=img_size//16,
                       min_holes=1, min_height=img_size//32, min_width=img_size//32, 
                       fill_value=0, p=0.3),
        
        # Normalize
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_valid_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class FrequencyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
    
    def forward(self, x):
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft)
        B, C, H, W = x.shape
        cy, cx = H // 2, W // 2
        y, xg = torch.meshgrid(torch.arange(H, device=x.device), 
                               torch.arange(W, device=x.device), indexing='ij')
        mask = (torch.sqrt((y - cy).float()**2 + (xg - cx).float()**2) > 15).float()
        mask = mask.unsqueeze(0).unsqueeze(0)
        img_back = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft_shift * mask)))
        return F.relu(self.bn(self.conv(img_back)))

class GraphModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gcn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        nodes = x.flatten(2).transpose(1, 2)  # B, HW, C
        q = self.proj(nodes)
        attn = F.softmax(torch.matmul(q, q.transpose(-2, -1)) / (C**0.5), dim=-1)
        out = self.gcn(torch.matmul(attn, nodes))
        return x + out.transpose(1, 2).reshape(B, C, H, W)

class TITAN_Model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Encoder - Swin has 4 stages with indices 0,1,2,3
        self.encoder = timm.create_model(
            CFG.MODEL_NAME, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )
        self.dims = self.encoder.feature_info.channels()
        
        # Physics-based frequency analysis
        self.physics = FrequencyBlock()
        
        # Graph module for spatial relationships
        self.graph = GraphModule(self.dims[-1])
        
        # Feature fusion
        self.fusion = nn.Conv2d(self.dims[-1] + 32, 256, 1)
        
        # Multi-scale decoder with skip connections
        self.decoder4 = self._decoder_block(256, 128)
        self.decoder3 = self._decoder_block(128 + self.dims[-2], 64)
        self.decoder2 = self._decoder_block(64 + self.dims[-3], 32)
        self.decoder1 = self._decoder_block(32 + self.dims[-4], 16)
        
        # Final prediction
        self.final = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        
        # Deep supervision heads
        self.ds4 = nn.Conv2d(128, 1, 1)
        self.ds3 = nn.Conv2d(64, 1, 1)
        self.ds2 = nn.Conv2d(32, 1, 1)
    
    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Encoder
        features = self.encoder(x)
        f1, f2, f3, f4 = features
        
        # Handle dimension ordering
        if f4.ndim == 4 and f4.shape[-1] == self.dims[-1]:
            f4 = f4.permute(0, 3, 1, 2)
        if f3.ndim == 4 and f3.shape[-1] == self.dims[-2]:
            f3 = f3.permute(0, 3, 1, 2)
        if f2.ndim == 4 and f2.shape[-1] == self.dims[-3]:
            f2 = f2.permute(0, 3, 1, 2)
        if f1.ndim == 4 and f1.shape[-1] == self.dims[-4]:
            f1 = f1.permute(0, 3, 1, 2)
        
        # Physics branch
        phys = self.physics(x)
        
        # Graph reasoning
        f4_graph = self.graph(f4)
        
        # Fusion
        phys_resized = F.interpolate(phys, size=f4_graph.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([f4_graph, phys_resized], dim=1))
        
        # Decoder with skip connections
        d4 = self.decoder4(fused)
        d4_up = F.interpolate(d4, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        
        d3 = self.decoder3(torch.cat([d4_up, f3], dim=1))
        d3_up = F.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        
        d2 = self.decoder2(torch.cat([d3_up, f2], dim=1))
        d2_up = F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        
        d1 = self.decoder1(torch.cat([d2_up, f1], dim=1))
        
        # Final output
        out = self.final(d1)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        if self.training:
            # Deep supervision
            ds4 = F.interpolate(self.ds4(d4), size=input_size, mode='bilinear', align_corners=False)
            ds3 = F.interpolate(self.ds3(d3), size=input_size, mode='bilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=input_size, mode='bilinear', align_corners=False)
            return out, ds4, ds3, ds2
        
        return out

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
    
    def forward(self, pred, target, ds_preds=None):
        loss = (CFG.DICE_WEIGHT * self.dice(pred, target) +
                CFG.BCE_WEIGHT * self.bce(pred, target) +
                CFG.FOCAL_WEIGHT * self.focal(pred, target))
        
        # Deep supervision loss
        if ds_preds is not None:
            for i, ds in enumerate(ds_preds):
                weight = 0.5 ** (i + 1)  # Decreasing weights
                loss += weight * (self.dice(ds, target) + self.bce(ds, target))
        
        return loss

# ============================================================================
# METRICS
# ============================================================================
def compute_dice(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-8) / (pred.sum() + target.sum() + 1e-8)

def compute_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    total_dice = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if CFG.USE_AMP:
            with autocast():
                outputs = model(images)
                if isinstance(outputs, tuple):
                    pred, ds4, ds3, ds2 = outputs
                    loss = criterion(pred, masks, [ds4, ds3, ds2])
                else:
                    pred = outputs
                    loss = criterion(pred, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, tuple):
                pred, ds4, ds3, ds2 = outputs
                loss = criterion(pred, masks, [ds4, ds3, ds2])
            else:
                pred = outputs
                loss = criterion(pred, masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # NOTE: Scheduler steps at epoch end, not here
        
        with torch.no_grad():
            dice = compute_dice(pred, masks)
        
        total_loss += loss.item()
        total_dice += dice.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})
    
    return total_loss / len(loader), total_dice / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    for images, masks in tqdm(loader, desc='Validating'):
        images = images.to(device)
        masks = masks.to(device)
        
        pred = model(images)
        loss = criterion(pred, masks)
        
        dice = compute_dice(pred, masks)
        iou = compute_iou(pred, masks)
        
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
    
    return total_loss / len(loader), total_dice / len(loader), total_iou / len(loader)

# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_data():
    """Prepare training data paths"""
    image_paths = []
    mask_paths = []
    
    # Forged images with masks
    forged_dir = CFG.TRAIN_IMAGES / 'forged'
    if forged_dir.exists():
        for img_path in forged_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
                # Find corresponding mask
                img_id = img_path.stem
                # Try different mask naming conventions
                mask_candidates = [
                    CFG.TRAIN_MASKS / f"{img_id}.npy",
                    CFG.TRAIN_MASKS / f"{img_id}.png",
                    CFG.TRAIN_MASKS / f"{img_id}_mask.png",
                ]
                mask_path = None
                for mc in mask_candidates:
                    if mc.exists():
                        mask_path = mc
                        break
                
                image_paths.append(img_path)
                mask_paths.append(mask_path)
    
    # Authentic images (no mask = all zeros)
    authentic_dir = CFG.TRAIN_IMAGES / 'authentic'
    if authentic_dir.exists():
        for img_path in authentic_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
                image_paths.append(img_path)
                mask_paths.append(None)  # No mask for authentic
    
    # Supplemental data
    if CFG.SUPP_IMAGES.exists():
        for img_path in CFG.SUPP_IMAGES.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.tif', '.tiff']:
                img_id = img_path.stem
                mask_path = CFG.SUPP_MASKS / f"{img_id}.npy"
                if mask_path.exists():
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
    
    print(f"📊 Total samples: {len(image_paths)}")
    print(f"   Forged: {sum(1 for m in mask_paths if m is not None)}")
    print(f"   Authentic: {sum(1 for m in mask_paths if m is None)}")
    
    return image_paths, mask_paths

# ============================================================================
# MAIN TRAINING
# ============================================================================
def train_fold(fold, train_idx, val_idx, image_paths, mask_paths):
    print(f"\n{'='*60}")
    print(f"🏋️ FOLD {fold}")
    print(f"{'='*60}")
    
    # Prepare datasets
    train_images = [image_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_images = [image_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]
    
    train_dataset = ForgeryDataset(train_images, train_masks, 
                                   get_train_transforms(CFG.IMG_SIZE), is_train=True)
    val_dataset = ForgeryDataset(val_images, val_masks,
                                 get_valid_transforms(CFG.IMG_SIZE), is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE,
                             shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE,
                           shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    
    # Model - Train from scratch with ImageNet pretrained backbone
    model = TITAN_Model(pretrained=True).to(CFG.DEVICE)
    print("✓ Using ImageNet pretrained Swin backbone (training decoder from scratch)")
    
    # Optimizer & Scheduler - ReduceLROnPlateau for stable training
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, 
                                  weight_decay=CFG.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, 
                                   min_lr=CFG.MIN_LR, verbose=True)
    
    # Loss
    criterion = CombinedLoss()
    
    # Mixed precision
    scaler = GradScaler() if CFG.USE_AMP else None
    
    # Training loop
    best_dice = 0
    patience_counter = 0
    
    for epoch in range(CFG.EPOCHS):
        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS}")
        
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, CFG.DEVICE
        )
        
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, CFG.DEVICE)
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Valid - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Step scheduler based on validation dice
        scheduler.step(val_dice)
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            save_path = CFG.OUTPUT_DIR / f'TITAN_08_fold{fold}_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model: {save_path} (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Save final model
    final_path = CFG.OUTPUT_DIR / f'TITAN_08_fold{fold}_final.pth'
    torch.save(model.state_dict(), final_path)
    
    print(f"\n✅ Fold {fold} complete - Best Dice: {best_dice:.4f}")
    return best_dice

def main():
    print("="*70)
    print("🚀 TITAN TRAIN 0.8 - Starting Training")
    print("="*70)
    
    # Prepare data
    image_paths, mask_paths = prepare_data()
    
    if len(image_paths) == 0:
        print("❌ No training data found!")
        return
    
    # Cross-validation
    kfold = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths)):
        if fold not in CFG.TRAIN_FOLDS:
            continue
        
        best_dice = train_fold(fold, train_idx, val_idx, image_paths, mask_paths)
        fold_scores.append(best_dice)
    
    # Summary
    print("\n" + "="*70)
    print("📊 TRAINING COMPLETE")
    print("="*70)
    for i, score in enumerate(fold_scores):
        print(f"   Fold {CFG.TRAIN_FOLDS[i]}: Dice = {score:.4f}")
    print(f"   Mean Dice: {np.mean(fold_scores):.4f}")
    print("="*70)

if __name__ == "__main__":
    main()

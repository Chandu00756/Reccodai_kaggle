"""
================================================================================
TITAN TRAIN 0.8: Ultimate Training Script for 0.8+ F1
================================================================================
TARGET: 0.8+ F1 Score - FULLY DEBUGGED VERSION

FIXES APPLIED:
1. ✅ Per-sample dice/IoU computation (was computing over flattened batch)
2. ✅ Removed broken FrequencyBlock and GraphModule (caused instability)
3. ✅ Disabled AMP for MPS (Mac doesn't support it well)
4. ✅ NUM_WORKERS=0 for Mac (avoids multiprocessing spawn issues)
5. ✅ Differential LR: 0.1x for encoder, 1x for decoder
6. ✅ Lighter deep supervision weights (0.1, 0.05 instead of 0.5, 0.25)
7. ✅ Added Dropout2d for regularization
8. ✅ Handle empty masks correctly in metrics
9. ✅ Conservative LR (5e-5) to avoid collapse
10. ✅ Save full checkpoint with optimizer state
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
    
    # Training - STABLE SETTINGS
    EPOCHS = 100
    BATCH_SIZE = 4
    NUM_WORKERS = 0         # 0 for Mac to avoid multiprocessing issues
    LEARNING_RATE = 5e-5    # Conservative LR
    MIN_LR = 1e-7
    WEIGHT_DECAY = 1e-4
    
    # Loss weights
    DICE_WEIGHT = 0.5
    BCE_WEIGHT = 0.3
    FOCAL_WEIGHT = 0.2
    
    # Cross-validation
    N_FOLDS = 5
    TRAIN_FOLDS = [0]
    
    # Early stopping
    PATIENCE = 20
    
    # Mixed precision - DISABLE for MPS (Mac)
    USE_AMP = False  # MPS doesn't support AMP well
    
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
    torch.backends.cudnn.benchmark = False  # Deterministic

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
            if image is None or image.size == 0:
                raise ValueError("Invalid image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
        except Exception:
            image = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)
            h, w = CFG.IMG_SIZE, CFG.IMG_SIZE
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = np.zeros((h, w), dtype=np.float32)
        
        if mask_path is not None and os.path.exists(str(mask_path)):
            try:
                if str(mask_path).endswith('.npy'):
                    loaded_mask = np.load(str(mask_path))
                else:
                    loaded_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if loaded_mask is not None and loaded_mask.size > 0:
                    # Ensure 2D
                    if len(loaded_mask.shape) == 3:
                        loaded_mask = loaded_mask[:, :, 0]
                    # Binarize
                    loaded_mask = (loaded_mask > 0).astype(np.float32)
                    # Resize to match image
                    if loaded_mask.shape[0] != h or loaded_mask.shape[1] != w:
                        loaded_mask = cv2.resize(loaded_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = loaded_mask
            except Exception as e:
                pass  # Keep zeros mask
        
        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask is [1, H, W] tensor
        if isinstance(mask, torch.Tensor):
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return image, mask

# ============================================================================
# AUGMENTATIONS
# ============================================================================
def get_train_transforms(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Light color augmentation
        A.OneOf([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        ], p=0.5),
        
        # Light noise
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        
        # Light blur
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # JPEG compression
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        
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
# MODEL ARCHITECTURE - SIMPLIFIED & STABLE
# ============================================================================
class TITAN_Model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Encoder
        self.encoder = timm.create_model(
            CFG.MODEL_NAME, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )
        self.dims = self.encoder.feature_info.channels()
        # dims = [128, 256, 512, 1024] for swin_base
        
        # Decoder - UNet style with skip connections
        self.decoder4 = self._decoder_block(self.dims[3], 256)
        self.decoder3 = self._decoder_block(256 + self.dims[2], 128)
        self.decoder2 = self._decoder_block(128 + self.dims[1], 64)
        self.decoder1 = self._decoder_block(64 + self.dims[0], 32)
        
        # Final prediction
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        # Deep supervision (lighter weights)
        self.ds3 = nn.Conv2d(128, 1, 1)
        self.ds2 = nn.Conv2d(64, 1, 1)
    
    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Encoder
        features = self.encoder(x)
        f1, f2, f3, f4 = features
        
        # Handle Swin output format (B, H, W, C) -> (B, C, H, W)
        if f4.ndim == 4 and f4.shape[-1] == self.dims[3]:
            f4 = f4.permute(0, 3, 1, 2).contiguous()
        if f3.ndim == 4 and f3.shape[-1] == self.dims[2]:
            f3 = f3.permute(0, 3, 1, 2).contiguous()
        if f2.ndim == 4 and f2.shape[-1] == self.dims[1]:
            f2 = f2.permute(0, 3, 1, 2).contiguous()
        if f1.ndim == 4 and f1.shape[-1] == self.dims[0]:
            f1 = f1.permute(0, 3, 1, 2).contiguous()
        
        # Decoder with skip connections
        d4 = self.decoder4(f4)
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
            # Deep supervision with LIGHT weights
            ds3 = F.interpolate(self.ds3(d3), size=input_size, mode='bilinear', align_corners=False)
            ds2 = F.interpolate(self.ds2(d2), size=input_size, mode='bilinear', align_corners=False)
            return out, ds3, ds2
        
        return out

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).contiguous()
        target = target.contiguous()
        
        # Compute per-sample dice then average
        batch_size = pred.shape[0]
        dice_sum = 0.0
        
        for i in range(batch_size):
            p = pred[i].reshape(-1)  # Use reshape instead of view for non-contiguous tensors
            t = target[i].reshape(-1)
            intersection = (p * t).sum()
            dice = (2. * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_sum += (1 - dice)
        
        return dice_sum / batch_size

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
        main_loss = (CFG.DICE_WEIGHT * self.dice(pred, target) +
                     CFG.BCE_WEIGHT * self.bce(pred, target) +
                     CFG.FOCAL_WEIGHT * self.focal(pred, target))
        
        # Deep supervision with VERY light weights
        if ds_preds is not None:
            ds_loss = 0.0
            for i, ds in enumerate(ds_preds):
                weight = 0.1 / (i + 1)  # 0.1, 0.05, etc.
                ds_loss += weight * self.dice(ds, target)
            main_loss = main_loss + ds_loss
        
        return main_loss

# ============================================================================
# METRICS - FIXED PER-SAMPLE COMPUTATION
# ============================================================================
def compute_dice(pred, target, threshold=0.5):
    """Compute dice score per sample and return mean"""
    pred = (torch.sigmoid(pred) > threshold).float()
    batch_size = pred.shape[0]
    
    dice_scores = []
    for i in range(batch_size):
        p = pred[i].reshape(-1)  # Use reshape instead of view
        t = target[i].reshape(-1)
        
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        
        if union < 1:  # Both empty - perfect match
            dice_scores.append(1.0)
        else:
            dice = (2. * intersection + 1e-8) / (union + 1e-8)
            dice_scores.append(dice.item())
    
    return np.mean(dice_scores)

def compute_iou(pred, target, threshold=0.5):
    """Compute IoU per sample and return mean"""
    pred = (torch.sigmoid(pred) > threshold).float()
    batch_size = pred.shape[0]
    
    iou_scores = []
    for i in range(batch_size):
        p = pred[i].reshape(-1)  # Use reshape instead of view
        t = target[i].reshape(-1)
        
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        
        if union < 1:  # Both empty
            iou_scores.append(1.0)
        else:
            iou = (intersection + 1e-8) / (union + 1e-8)
            iou_scores.append(iou.item())
    
    return np.mean(iou_scores)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        if isinstance(outputs, tuple):
            pred = outputs[0]
            ds_preds = outputs[1:]
            loss = criterion(pred, masks, ds_preds)
        else:
            pred = outputs
            loss = criterion(pred, masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            dice = compute_dice(pred, masks)
        
        total_loss += loss.item()
        total_dice += dice
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    return total_loss / num_batches, total_dice / num_batches

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    num_batches = 0
    
    for images, masks in tqdm(loader, desc='Validating'):
        images = images.to(device)
        masks = masks.to(device)
        
        pred = model(images)
        loss = criterion(pred, masks)
        
        dice = compute_dice(pred, masks)
        iou = compute_iou(pred, masks)
        
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        num_batches += 1
    
    return total_loss / num_batches, total_dice / num_batches, total_iou / num_batches

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
                img_id = img_path.stem
                mask_candidates = [
                    CFG.TRAIN_MASKS / f"{img_id}.npy",
                    CFG.TRAIN_MASKS / f"{img_id}.png",
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
                mask_paths.append(None)
    
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
                             shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE,
                           shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=False)
    
    # Model
    print("📦 Loading model...")
    model = TITAN_Model(pretrained=True).to(CFG.DEVICE)
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer with different LR for encoder vs decoder
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': CFG.LEARNING_RATE * 0.1},  # Lower LR for pretrained
        {'params': decoder_params, 'lr': CFG.LEARNING_RATE},
    ], weight_decay=CFG.WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, 
                                   min_lr=CFG.MIN_LR)
    
    # Loss
    criterion = CombinedLoss()
    
    # Scaler for AMP (not used on MPS)
    scaler = None
    
    # Training loop
    best_dice = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    
    for epoch in range(CFG.EPOCHS):
        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS}")
        
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, CFG.DEVICE
        )
        
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, CFG.DEVICE)
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Valid - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Step scheduler
        scheduler.step(val_dice)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            save_path = CFG.OUTPUT_DIR / f'TITAN_08_fold{fold}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
            }, save_path)
            print(f"💾 Saved best model: {save_path} (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{CFG.PATIENCE})")
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
    print(f"Device: {CFG.DEVICE}")
    print(f"Image Size: {CFG.IMG_SIZE}")
    print(f"Batch Size: {CFG.BATCH_SIZE}")
    print(f"Learning Rate: {CFG.LEARNING_RATE}")
    print(f"Epochs: {CFG.EPOCHS}")
    print("="*70)
    
    # Prepare data
    image_paths, mask_paths = prepare_data()
    
    if len(image_paths) == 0:
        print("❌ No training data found!")
        return
    
    # Verify some data
    print("\n🔍 Verifying data...")
    valid_count = 0
    for i in range(min(5, len(image_paths))):
        img = cv2.imread(str(image_paths[i]))
        if img is not None:
            valid_count += 1
    print(f"   Verified {valid_count}/5 sample images")
    
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
    if fold_scores:
        print(f"   Mean Dice: {np.mean(fold_scores):.4f}")
    print("="*70)

if __name__ == "__main__":
    main()

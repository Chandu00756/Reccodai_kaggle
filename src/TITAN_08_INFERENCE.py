"""
================================================================================
TITAN 0.8 INFERENCE - Kaggle Submission Script
================================================================================
Uses weights trained with TITAN_TRAIN_08.py locally
Runs inference on Kaggle test set
================================================================================
"""

import os
import sys
import cv2
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

try:
    import timm
except ImportError:
    os.system('pip install timm -q')
    import timm

# ============================================================================
# KAGGLE PATHS - DO NOT CHANGE
# ============================================================================
DATA_ROOT = Path('/kaggle/input/recodai-luc-scientific-image-forgery-detection')
TEST_IMAGES = DATA_ROOT / 'test_images'
SAMPLE_SUB = DATA_ROOT / 'sample_submission.csv'

# Model weights - upload your trained model to Kaggle dataset
WEIGHTS_PATH = Path('/kaggle/input/titan08-weights/TITAN_08_fold0_best.pth')

# ============================================================================
# INFERENCE CONFIG
# ============================================================================
MODEL_NAME = 'swin_base_patch4_window12_384'
IMG_SIZE = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Thresholds - OPTIMIZED for F1 (more aggressive detection)
HIGH_THRESH = 0.35      # Lower = more detections
LOW_THRESH = 0.20       # Catch weaker signals
MIN_PIXELS = 200        # Allow smaller forgeries
MIN_STRONG = 10         # More lenient

print(f"🚀 TITAN 0.8 Inference - Device: {DEVICE}")

# ============================================================================
# MODEL ARCHITECTURE - MUST MATCH TITAN_TRAIN_08.py EXACTLY
# ============================================================================
class TITAN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = timm.create_model(
            MODEL_NAME, 
            pretrained=False,
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
        
        return out

# ============================================================================
# RLE ENCODING - EXACT COMPETITION FORMAT
# ============================================================================
def rle_encode(mask):
    """Encode mask to RLE format required by competition"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return json.dumps([int(x) for x in runs])

# ============================================================================
# PREPROCESSING
# ============================================================================
def preprocess_image(image_path):
    """Load and preprocess image for inference"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    
    original_shape = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    
    return img, original_shape

# ============================================================================
# INFERENCE WITH TTA
# ============================================================================
@torch.no_grad()
def predict_with_tta(model, img_tensor, original_shape):
    """Predict with test-time augmentation"""
    model.eval()
    
    # Original
    pred = torch.sigmoid(model(img_tensor.to(DEVICE)))
    
    # Horizontal flip
    img_hflip = torch.flip(img_tensor, dims=[3])
    pred_hflip = torch.sigmoid(model(img_hflip.to(DEVICE)))
    pred_hflip = torch.flip(pred_hflip, dims=[3])
    
    # Vertical flip
    img_vflip = torch.flip(img_tensor, dims=[2])
    pred_vflip = torch.sigmoid(model(img_vflip.to(DEVICE)))
    pred_vflip = torch.flip(pred_vflip, dims=[2])
    
    # Average predictions
    pred_avg = (pred + pred_hflip + pred_vflip) / 3.0
    
    # Convert to numpy and resize to original
    pred_np = pred_avg.squeeze().cpu().numpy()
    pred_resized = cv2.resize(pred_np, (original_shape[1], original_shape[0]))
    
    return pred_resized

# ============================================================================
# MAIN INFERENCE
# ============================================================================
def main():
    print("="*70)
    print("🚀 TITAN 0.8 INFERENCE")
    print("="*70)
    
    # Load model
    print("📦 Loading model...")
    model = TITAN_Model().to(DEVICE)
    
    if WEIGHTS_PATH.exists():
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
        # Handle both direct state_dict and checkpoint dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=True)
        print(f"✓ Loaded weights from {WEIGHTS_PATH}")
    else:
        print(f"❌ ERROR: Weights not found at {WEIGHTS_PATH}")
        print("   Please upload your trained weights to Kaggle dataset")
        return
    
    model.eval()
    
    # Load sample submission for correct format
    sample_sub = pd.read_csv(SAMPLE_SUB)
    print(f"📋 Sample submission: {len(sample_sub)} rows")
    
    # Get test images
    test_images = sorted(glob(str(TEST_IMAGES / '*')))
    print(f"🖼️ Found {len(test_images)} test images")
    
    # Inference
    results = []
    
    for img_path in tqdm(test_images, desc="Predicting"):
        img_path = Path(img_path)
        case_id = int(img_path.stem)  # Convert to int to match sample_sub
        
        # Preprocess
        img_tensor, original_shape = preprocess_image(img_path)
        
        if img_tensor is None:
            # Failed to load - mark as authentic
            results.append({
                'case_id': case_id,
                'annotation': 'authentic'
            })
            continue
        
        # Predict with TTA
        pred = predict_with_tta(model, img_tensor, original_shape)
        
        # Apply hysteresis thresholding with connected components
        strong_mask = (pred >= HIGH_THRESH).astype(np.uint8)
        weak_mask = (pred >= LOW_THRESH).astype(np.uint8)
        
        # Check if we have enough strong pixels globally
        if strong_mask.sum() < MIN_STRONG:
            results.append({
                'case_id': case_id,
                'annotation': 'authentic'
            })
            continue
        
        # Find connected components in weak mask
        num_labels, labels = cv2.connectedComponents(weak_mask, connectivity=8)
        final_mask = np.zeros_like(weak_mask)
        
        # Keep components that contain strong pixels
        for i in range(1, num_labels):
            component_mask = (labels == i).astype(np.uint8)
            # Check if this component has strong pixels
            strong_in_component = cv2.bitwise_and(component_mask, strong_mask).sum()
            
            if strong_in_component >= MIN_STRONG and component_mask.sum() >= MIN_PIXELS:
                final_mask = cv2.bitwise_or(final_mask, component_mask)
        
        # Check final mask
        if final_mask.sum() < MIN_PIXELS:
            results.append({
                'case_id': case_id,
                'annotation': 'authentic'
            })
            continue
        
        # Encode to RLE
        rle = rle_encode(final_mask)
            
        results.append({
            'case_id': case_id,
            'annotation': rle
        })
    
    # Create submission dataframe
    submission = pd.DataFrame(results)
    
    # Merge with sample submission to ensure correct order and all rows
    submission = sample_sub[['case_id']].merge(submission, on='case_id', how='left')
    submission['annotation'] = submission['annotation'].fillna('authentic')
    
    # Save
    submission.to_csv('submission.csv', index=False)
    
    # Stats
    authentic_count = (submission['annotation'] == 'authentic').sum()
    forged_count = len(submission) - authentic_count
    print(f"\n📊 Submission Stats:")
    print(f"   Total: {len(submission)}")
    print(f"   Authentic: {authentic_count} ({100*authentic_count/len(submission):.1f}%)")
    print(f"   Forged: {forged_count} ({100*forged_count/len(submission):.1f}%)")
    print(f"✅ Saved to submission.csv")
    print(f"\n💡 Tip: If F1 is still low, try even lower thresholds (0.25/0.15)")

if __name__ == "__main__":
    main()
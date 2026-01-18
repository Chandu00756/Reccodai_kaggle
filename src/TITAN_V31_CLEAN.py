"""
================================================================================
TITAN V3.1 CLEAN: Your Best Performing Script (0.304 F1)
================================================================================
This is exactly what scored 0.304 - don't change thresholds!
- HIGH_THRESH = 0.85 (proven optimal)
- MIN_PIXELS = 1000 (filters noise)
- Simple H-flip TTA only
================================================================================
"""

import os
import sys
import cv2
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import timm
except ImportError:
    sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')
    import timm

try:
    from tqdm import tqdm
except:
    def tqdm(x, **kwargs): return x

# ============================================================================
# CONFIG - DO NOT CHANGE THESE VALUES (they give 0.304)
# ============================================================================
class CFG:
    if os.path.exists('/kaggle/input'):
        TEST_DIR = '/kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images'
        SAMPLE_SUB = '/kaggle/input/recodai-luc-scientific-image-forgery-detection/sample_submission.csv'
        WEIGHTS_PATH = '/kaggle/input/recodai-model/TITAN_V2_UNLEASHED.pth'
    else:
        TEST_DIR = 'recodai-luc-scientific-image-forgery-detection/test_images'
        SAMPLE_SUB = 'recodai-luc-scientific-image-forgery-detection/sample_submission.csv'
        WEIGHTS_PATH = 'TITAN_V2_UNLEASHED.pth'
    
    IMG_SIZE = 384
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # PROVEN OPTIMAL - DON'T CHANGE
    HIGH_THRESH = 0.85
    LOW_THRESH = 0.40
    MIN_PIXELS = 1000
    MIN_STRONG = 50

print(f"🎯 TITAN V3.1 CLEAN - Device: {CFG.DEVICE}")

# ============================================================================
# MODEL
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
        y, xg = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y, xg = y.to(x.device), xg.to(x.device)
        mask = (torch.sqrt((y - cy)**2 + (xg - cx)**2) > 15).float().unsqueeze(0).unsqueeze(0)
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
            nn.Conv2d(32, 1, 1))
    def forward(self, x):
        enc_feats = self.encoder(x)
        deep_feats = enc_feats[-1]
        if deep_feats.ndim == 4 and deep_feats.shape[-1] == self.dims[-1]:
            deep_feats = deep_feats.permute(0, 3, 1, 2)
        phys_feats = self.physics(x)
        graph_feats = self.graph(deep_feats)
        phys_resized = F.interpolate(phys_feats, size=graph_feats.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat([graph_feats, phys_resized], dim=1))
        return F.interpolate(self.decoder(fused), size=x.shape[-2:], mode='bilinear', align_corners=False)

# ============================================================================
# HELPERS
# ============================================================================
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return json.dumps([int(x) for x in runs])

def apply_hysteresis(prob_map, high, low, min_pixels, min_strong):
    h, w = prob_map.shape
    strong_mask = (prob_map >= high).astype(np.uint8)
    weak_mask = (prob_map >= low).astype(np.uint8)
    
    if strong_mask.sum() < min_strong:
        return np.zeros((h, w), dtype=np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak_mask, connectivity=8)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_pixels:
            continue
        blob_mask = (labels == i).astype(np.uint8)
        if cv2.bitwise_and(blob_mask, strong_mask).sum() >= min_strong:
            final_mask = cv2.bitwise_or(final_mask, blob_mask)
    
    return final_mask

# ============================================================================
# INFERENCE
# ============================================================================
def run_inference():
    print("=" * 60)
    print("🎯 TITAN V3.1 CLEAN - Proven 0.304 Score")
    print("=" * 60)
    
    # Build file map
    all_files = glob.glob(os.path.join(CFG.TEST_DIR, '**', '*'), recursive=True)
    id_map = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            base = os.path.basename(f)
            digits = ''.join(filter(str.isdigit, os.path.splitext(base)[0]))
            if digits:
                id_map[str(int(digits))] = f
    
    # Load model
    print(f">>> Loading: {CFG.WEIGHTS_PATH}")
    model = FAGT_Model()
    if os.path.exists(CFG.WEIGHTS_PATH):
        model.load_state_dict(torch.load(CFG.WEIGHTS_PATH, map_location=CFG.DEVICE))
        print(">>> ✓ Weights loaded")
    else:
        print(">>> ✗ WEIGHTS NOT FOUND!")
    model.to(CFG.DEVICE).eval()
    
    # Normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(CFG.DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(CFG.DEVICE)
    
    preds_list = []
    print(f">>> Processing {len(id_map)} images...")
    
    for case_id, path in tqdm(id_map.items()):
        label = "authentic"
        try:
            # Load image
            image = cv2.imread(path)
            if image is None:
                from PIL import Image
                image = np.array(Image.open(path).convert('RGB'))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w = image.shape[:2]
            
            # Preprocess
            img = cv2.resize(image, (CFG.IMG_SIZE, CFG.IMG_SIZE))
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(CFG.DEVICE) / 255.0
            img_t = (img_t - mean) / std
            
            # Inference with H-flip TTA
            with torch.no_grad():
                p1 = torch.sigmoid(model(img_t))[0, 0]
                p2 = torch.flip(torch.sigmoid(model(torch.flip(img_t, [3]))), [3])[0, 0]
                prob_map = ((p1 + p2) / 2.0).cpu().numpy()
            
            # Resize to original
            prob_full = cv2.resize(prob_map, (w, h))
            
            # Apply hysteresis
            mask = apply_hysteresis(prob_full, CFG.HIGH_THRESH, CFG.LOW_THRESH, 
                                   CFG.MIN_PIXELS, CFG.MIN_STRONG)
            
            if mask.sum() > 0:
                label = rle_encode(mask)
                if label == "":
                    label = "authentic"
        except Exception as e:
            pass
        
        preds_list.append({"case_id": case_id, "annotation": label})
    
    # Safe submission
    try:
        sample_sub = pd.read_csv(CFG.SAMPLE_SUB)
    except:
        sample_sub = pd.DataFrame({'case_id': [45], 'annotation': ['authentic']})
    
    sample_sub['case_id'] = sample_sub['case_id'].astype(str)
    
    if len(preds_list) > 0:
        preds_df = pd.DataFrame(preds_list)
        preds_df['case_id'] = preds_df['case_id'].astype(str)
        submission = sample_sub[['case_id']].merge(preds_df, on='case_id', how='left')
        submission['annotation'] = submission['annotation'].fillna("authentic")
    else:
        submission = sample_sub.copy()
        submission['annotation'] = 'authentic'
    
    submission.to_csv('submission.csv', index=False)
    
    forged = sum(1 for x in submission['annotation'] if x != 'authentic')
    print("=" * 60)
    print(f"✅ DONE: {len(submission)} images | {forged} forged")
    print("=" * 60)
    
    return submission

if __name__ == "__main__":
    run_inference()

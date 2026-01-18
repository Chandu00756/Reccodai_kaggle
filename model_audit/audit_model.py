# =============================================================================
# FAGT FINAL AUDIT: THE "HIDDEN SET SIMULATOR" (FIXED)
# =============================================================================
import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
import timm
import torch.nn as nn
import torch.nn.functional as F  # <--- Added this missing import
import torchvision.transforms as T
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# --- CONFIGURATION ---
# UPDATE THESE PATHS IF NEEDED
REAL_DATA_ROOT = "/Users/chanduchitikam/recodai/recodai-luc-scientific-image-forgery-detection"
IMG_DIR = os.path.join(REAL_DATA_ROOT, "train_images/forged")
MASK_DIR = os.path.join(REAL_DATA_ROOT, "train_masks")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "weights", "FAGT_best_fold0.pth")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- REPLICATE THE SPLIT ---
def get_validation_images():
    # 1. List all forged images
    all_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))
    if not all_files:
        print(f"ERROR: No images found in {IMG_DIR}")
        return []
        
    df = pd.DataFrame({'image_path': all_files})
    
    # 2. Re-create the KFold split with seed 42
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Get the indices for Fold 0 Validation
    train_idx, val_idx = next(kf.split(df))
    
    # 4. Return the paths corresponding to Validation Index
    val_files = df.iloc[val_idx]['image_path'].tolist()
    print(f">>> Identified {len(val_files)} Unseen Validation Images (Hidden Set Proxy)")
    return val_files

# =============================================================================
# 1. MODEL ARCHITECTURE
# =============================================================================
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
        return x + self.gcn(torch.matmul(attn, nodes)).transpose(1, 2).reshape(B, C, H, W)

class FAGT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('swin_base_patch4_window12_384', pretrained=False, features_only=True)
        self.dims = self.encoder.feature_info.channels()
        self.physics = FrequencyBlock()
        self.graph = GraphModule(self.dims[-1])
        self.fusion = nn.Conv2d(self.dims[-1] + 32, 256, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        enc_feats = self.encoder(x)
        deep_feats = enc_feats[-1]
        if deep_feats.ndim == 4 and deep_feats.shape[-1] == self.dims[-1]:
            deep_feats = deep_feats.permute(0, 3, 1, 2)
        phys_feats = self.physics(x)
        graph_feats = self.graph(deep_feats)
        phys_resized = F.interpolate(phys_feats, size=graph_feats.shape[-2:], mode='bilinear')
        fused = self.fusion(torch.cat([graph_feats, phys_resized], dim=1))
        logits = self.decoder(fused)
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear')

# =============================================================================
# 2. EVALUATION LOGIC
# =============================================================================
def load_model():
    print(f">>> Loading Model on {DEVICE}...")
    model = FAGT_Model().to(DEVICE)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(">>> Weights Loaded Successfully.")
    except Exception as e:
        print(f"ERROR: Could not load weights. {e}")
        return None
    model.eval()
    return model

def get_probability_map(model, img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (384, 384))
    
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_base = torch.sigmoid(model(img_tensor))
        pred_flip = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [3]))), [3])
        avg_pred = (pred_base + pred_flip) / 2.0
    return avg_pred.cpu().numpy()[0,0]

def evaluate():
    model = load_model()
    if model is None: return

    # USE ONLY UNSEEN IMAGES
    images = get_validation_images()
    
    if len(images) == 0:
        return

    print(f"\n>>> Simulating Hidden Test Set on {len(images)} unseen forgeries...")
    
    all_probs = []
    all_gts = []
    
    for img_path in tqdm(images):
        prob_map = get_probability_map(model, img_path)
        if prob_map is None: continue
        
        name = os.path.basename(img_path)
        mask_path = os.path.join(MASK_DIR, name.split('.')[0] + ".npy")
        
        gt_mask = None
        if os.path.exists(mask_path):
            try:
                raw = np.load(mask_path)
                if raw.ndim > 2: gt_mask = np.max(raw, axis=2)
                else: gt_mask = raw
            except: pass
            
        if gt_mask is None: gt_mask = np.zeros((384, 384), dtype=np.uint8)
        else: gt_mask = cv2.resize(gt_mask, (384, 384), interpolation=cv2.INTER_NEAREST)
            
        all_probs.append(prob_map.flatten())
        all_gts.append((gt_mask > 0).astype(np.uint8).flatten())

    y_scores = np.concatenate(all_probs)
    y_true = np.concatenate(all_gts)
    
    print("\n" + "="*60)
    print(f"{'PERCENTILE':<12} | {'F1 SCORE':<10} | {'PRECISION':<10} | {'RECALL':<10}")
    print("-" * 60)
    
    best_f1 = 0
    best_p = 0
    
    # Sweep to find the setting that maximizes score on UNSEEN data
    for p in [70, 75, 80, 85, 90, 95]:
        thresh = np.percentile(y_scores, p)
        y_pred = (y_scores > thresh).astype(np.uint8)
        
        f1 = f1_score(y_true, y_pred, zero_division=1)
        prec = precision_score(y_true, y_pred, zero_division=1)
        rec = recall_score(y_true, y_pred, zero_division=1)
        
        print(f"Top {100-p:>4.1f}%   | {f1:.4f}     | {prec:.4f}     | {rec:.4f} (Thresh: {thresh:.4f})")
        
        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            
    print("="*60)
    print(f">>> PREDICTED LEADERBOARD SCORE: {best_f1:.4f}")
    print(f">>> OPTIMAL SETTING: Keep Top {100-best_p:.1f}% of pixels")

if __name__ == "__main__":
    evaluate()
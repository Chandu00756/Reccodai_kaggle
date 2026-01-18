# =============================================================================
# NEW MODEL AUDIT: SWIN-LARGE + BAYARCONV (RIGOROUS TEST)
# =============================================================================
import os
import cv2
import glob
import torch
import numpy as np
import pandas as pd
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# --- CONFIGURATION ---
# UPDATE THESE PATHS TO MATCH YOUR LOCAL MAC SETUP
REAL_DATA_ROOT = "/Users/chanduchitikam/recodai/recodai-luc-scientific-image-forgery-detection"
IMG_DIR = os.path.join(REAL_DATA_ROOT, "train_images/forged")
MASK_DIR = os.path.join(REAL_DATA_ROOT, "train_masks")

# PATH TO YOUR NEW TRAINED WEIGHTS
WEIGHTS_PATH = "FAGT_Upgraded_Fold0.pth" # Ensure this file exists in the same folder

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = 'swin_large_patch4_window12_384.ms_in22k'

# --- 1. NEW MODEL ARCHITECTURE (MUST MATCH TRAINING EXACTLY) ---
class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        super(BayarConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.minus1 = (torch.ones(in_channels, out_channels, 1) * -1.000)
    def forward(self, x):
        self.conv.weight.data[:, :, self.kernel_size//2, self.kernel_size//2] = -1.0
        return self.conv(x)

class Upgraded_FAGT_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone
        self.encoder = timm.create_model(MODEL_NAME, pretrained=False, features_only=True)
        self.enc_channels = self.encoder.feature_info.channels()
        
        # Forensic
        self.bayar = BayarConv2d(3, 3, kernel_size=5)
        self.srm_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        # Fusion
        self.fusion = nn.Conv2d(self.enc_channels[-1] + 64, 512, 1)
        
        # Decoder
        self.up1_conv = nn.Conv2d(512, 256, 3, padding=1)
        self.up1_bn = nn.BatchNorm2d(256)
        
        self.up2_conv = nn.Conv2d(256 + self.enc_channels[-2], 128, 3, padding=1)
        self.up2_bn = nn.BatchNorm2d(128)
        
        self.up3_conv = nn.Conv2d(128 + self.enc_channels[-3], 64, 3, padding=1)
        self.up3_bn = nn.BatchNorm2d(64)
        
        self.up4_conv = nn.Conv2d(64, 32, 3, padding=1)
        self.up4_bn = nn.BatchNorm2d(32)
        
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        noise = self.bayar(x)
        noise_feat = self.srm_conv(noise)
        features = self.encoder(x)
        c2, c3, c4 = features[-3], features[-2], features[-1]
        
        if c4.shape[-1] == self.enc_channels[-1]: c4 = c4.permute(0, 3, 1, 2)
        if c3.shape[-1] == self.enc_channels[-2]: c3 = c3.permute(0, 3, 1, 2)
        if c2.shape[-1] == self.enc_channels[-3]: c2 = c2.permute(0, 3, 1, 2)

        noise_resized = F.interpolate(noise_feat, size=c4.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([c4, noise_resized], dim=1)
        x_dec = self.fusion(fused)
        
        x_dec = self.up1_conv(x_dec)
        x_dec = F.relu(self.up1_bn(x_dec))
        x_dec = F.interpolate(x_dec, scale_factor=2, mode='bilinear')
        
        if x_dec.shape[2:] != c3.shape[2:]: x_dec = F.interpolate(x_dec, size=c3.shape[2:])
        x_dec = torch.cat([x_dec, c3], dim=1)
        x_dec = self.up2_conv(x_dec)
        x_dec = F.relu(self.up2_bn(x_dec))
        x_dec = F.interpolate(x_dec, scale_factor=2, mode='bilinear')
        
        if x_dec.shape[2:] != c2.shape[2:]: x_dec = F.interpolate(x_dec, size=c2.shape[2:])
        x_dec = torch.cat([x_dec, c2], dim=1)
        x_dec = self.up3_conv(x_dec)
        x_dec = F.relu(self.up3_bn(x_dec))
        x_dec = F.interpolate(x_dec, scale_factor=2, mode='bilinear')
        
        x_dec = self.up4_conv(x_dec)
        x_dec = F.relu(self.up4_bn(x_dec))
        
        x_dec = F.interpolate(x_dec, size=(x.shape[2], x.shape[3]), mode='bilinear')
        return self.final(x_dec)

# --- 2. DATA SPLIT (UNSEEN IMAGES ONLY) ---
def get_validation_images():
    all_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))
    if not all_files:
        print(f"ERROR: No images found in {IMG_DIR}")
        return []
    
    # Same seed as training to ensure we pick the Validation set
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(all_files))
    
    val_files = [all_files[i] for i in val_idx]
    print(f">>> Identified {len(val_files)} Unseen Validation Images (Hidden Set Proxy)")
    return val_files

# --- 3. EVALUATION LOGIC ---
def load_model():
    print(f">>> Loading New Model on {DEVICE}...")
    model = Upgraded_FAGT_Model().to(DEVICE)
    if os.path.exists(WEIGHTS_PATH):
        try:
            state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
            model.load_state_dict(state, strict=True)
            print(">>> Weights Loaded Successfully (Strict Mode Passed).")
        except Exception as e:
            print(f"!!! LOAD FAILED: {e}")
            return None
    else:
        print(f"!!! ERROR: Weights file {WEIGHTS_PATH} not found.")
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
        # 4-Way TTA for robust scoring
        p1 = torch.sigmoid(model(img_tensor))
        p2 = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [3]))), [3]) # HFlip
        p3 = torch.flip(torch.sigmoid(model(torch.flip(img_tensor, [2]))), [2]) # VFlip
        avg_pred = (p1 + p2 + p3) / 3.0
        
    return avg_pred.cpu().numpy()[0,0]

def evaluate():
    model = load_model()
    if model is None: return

    images = get_validation_images()
    if len(images) == 0: return

    print(f"\n>>> Running Rigorous Audit on New Architecture...")
    
    all_probs = []
    all_gts = []
    
    for img_path in tqdm(images):
        prob_map = get_probability_map(model, img_path)
        if prob_map is None: continue
        
        name = os.path.basename(img_path)
        mask_path = os.path.join(MASK_DIR, name.split('.')[0] + ".npy")
        if not os.path.exists(mask_path):
             mask_path = os.path.join(MASK_DIR, name.split('.')[0] + "_mask.npy")
        
        gt_mask = None
        if os.path.exists(mask_path):
            try:
                raw = np.load(mask_path)
                if raw.ndim > 2: gt_mask = np.max(raw, axis=2)
                else: gt_mask = raw
                gt_mask = cv2.resize((gt_mask > 0).astype(np.uint8), (384, 384), interpolation=cv2.INTER_NEAREST)
            except: pass
            
        if gt_mask is None: gt_mask = np.zeros((384, 384), dtype=np.uint8)
            
        all_probs.append(prob_map.flatten())
        all_gts.append(gt_mask.flatten())

    y_scores = np.concatenate(all_probs)
    y_true = np.concatenate(all_gts)
    
    print("\n" + "="*70)
    print(f"{'PERCENTILE':<12} | {'F1 SCORE':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'THRESH':<10}")
    print("-" * 70)
    
    best_f1 = 0
    best_thresh = 0
    
    # Scan percentiles to find the 'Sweet Spot'
    for p in [85, 90, 92, 94, 95, 96, 97, 98, 99]:
        thresh = np.percentile(y_scores, p)
        y_pred = (y_scores > thresh).astype(np.uint8)
        
        f1 = f1_score(y_true, y_pred, zero_division=1)
        prec = precision_score(y_true, y_pred, zero_division=1)
        rec = recall_score(y_true, y_pred, zero_division=1)
        
        print(f"Top {100-p:>4.1f}%   | {f1:.4f}     | {prec:.4f}     | {rec:.4f}     | {thresh:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print("="*70)
    print(f">>> MAX F1 SCORE: {best_f1:.4f}")
    print(f">>> OPTIMAL THRESHOLD: {best_thresh:.4f}")
    print("If this is < 0.65, DO NOT SUBMIT. Train more.")

if __name__ == "__main__":
    evaluate()
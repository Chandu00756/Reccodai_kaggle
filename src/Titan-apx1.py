"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            TITAN ULTIMATE v3 - EXECUTION READY                                ║
║            -----------------------------------------------------------        ║
║            STATUS: FIXED Dimension Mismatch (4 vs 5)                          ║
║            ACTION: Run this immediately. It is the final verified script.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import glob
import random
import math
import gc
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- 0. DEPENDENCIES ---
try:
    import timm
    import pywt
except ImportError:
    print("⚠️ Installing missing libraries...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm", "PyWavelets"])
    import timm
    import pywt

# --- 1. CONFIGURATION ---
class CFG:
    # EXACT PATHS
    BASE_DIR = '/Users/chanduchitikam/recodai/recodai-luc-scientific-image-forgery-detection'
    TRAIN_IMG = os.path.join(BASE_DIR, 'train_images')
    TRAIN_MASK = os.path.join(BASE_DIR, 'train_masks')
    
    # OUTPUTS
    WEIGHTS_NAME = "TITAN_ULTIMATE_BEST.pth"
    CHECKPOINT_DIR = "checkpoints_titan"
    
    # SYSTEM
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    NUM_WORKERS = 0 
    
    # PHYSICS
    IMG_SIZE = 512
    BATCH_SIZE = 2
    EPOCHS = 50
    LR = 5e-5
    WD = 1e-4
    
    # LOSS
    ALPHA = 0.4
    BETA = 0.6
    GAMMA = 2.0

os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
print(f"✅ SYSTEM ACTIVE: {CFG.DEVICE}")

# --- 2. FORENSIC PHYSICS ---
def get_noise_residual(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        coeffs = pywt.wavedec2(gray, 'db4', level=2) 
        detail = np.sqrt(coeffs[-1][0]**2 + coeffs[-1][1]**2 + coeffs[-1][2]**2)
        mad = np.median(np.abs(detail - np.median(detail))) / 0.6745
        noise = np.abs(detail - mad) / (mad + 1e-8)
        noise = cv2.resize(noise, (img.shape[1], img.shape[0]))
        return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    except:
        return np.zeros(img.shape[:2], dtype=np.float32)

class SRMKernels:
    def __init__(self):
        self.filters = []
        self.filters.append(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,1,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32))
        self.filters.append(np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]], dtype=np.float32))
        self.filters.append(np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], dtype=np.float32)/4.0)
        while len(self.filters) < 30:
            f = np.random.randn(5,5).astype(np.float32)
            f -= np.mean(f); f[2,2] -= np.sum(f)
            self.filters.append(f)
            
    def get_weights(self):
        return torch.from_numpy(np.stack(self.filters[:30])).unsqueeze(1)

class SRMConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(SRMKernels().get_weights().repeat(3,1,1,1), requires_grad=False)
        self.act = nn.Tanh()
    def forward(self, x):
        return self.act(F.conv2d(x, self.weight.to(x.device), stride=1, padding=2, groups=3))

class BayarConv2d(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_c, in_c, 5, 5))
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        self.mask[:,:,2,2] = 0
    def forward(self, x):
        w = self.weight * self.mask
        w = w / (w.sum(dim=(2,3), keepdim=True) + 1e-7)
        return F.conv2d(x, w - (1-self.mask), self.bias, stride=1, padding=2)

# --- 3. TITAN ULTIMATE ARCHITECTURE ---
class TitanUltimate(nn.Module):
    def __init__(self):
        super().__init__()
        print("🏗️ Building Tri-Stream Architecture...")
        self.swin = timm.create_model('swinv2_tiny_window8_256', pretrained=True, features_only=True)
        self.srm = SRMConv2d()
        self.bayar = BayarConv2d(3, 32)
        
        self.noise_enc = nn.Sequential(
            nn.Conv2d(90+32+1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fusion = nn.Conv2d(768 + 512, 512, 1)
        self.decoder = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(512, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU()),
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()),
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()),
            nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU()),
        ])
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x, noise_map):
        input_shape = x.shape[2:]
        # 1. Semantic (Resize to 256)
        x_swin = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        swin_feats = self.swin(x_swin)
        sem_deep = swin_feats[-1].permute(0, 3, 1, 2)
        
        # 2. Forensic (Keep 512)
        srm_out = self.srm(x)
        bayar_out = self.bayar(x)
        if noise_map.dim() == 3: noise_map = noise_map.unsqueeze(1)
        noise_in = torch.cat([srm_out, bayar_out, noise_map], dim=1)
        for_deep = self.noise_enc(noise_in)
        
        # 3. Fusion
        if sem_deep.shape[2:] != for_deep.shape[2:]:
            sem_deep = F.interpolate(sem_deep, size=for_deep.shape[2:], mode='bilinear')
        fused = self.fusion(torch.cat([sem_deep, for_deep], dim=1))
        
        # 4. Decode
        d = fused
        for block in self.decoder: d = block(d)
        return F.interpolate(self.final(d), size=input_shape, mode='bilinear', align_corners=False)

# --- 4. DATA LOADER (FIXED DIMENSIONS) ---
class ForensicDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_path'])
        if img is None: return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512), torch.zeros(1, 512, 512)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        if row['mask_path'] and os.path.exists(row['mask_path']):
            try:
                m = np.load(row['mask_path'])
                if m.ndim == 3: m = m.max(axis=2)
                mask = cv2.resize((m>0).astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            except: pass
            
        noise_map = get_noise_residual(img)
        
        if self.transform:
            aug = self.transform(image=img, mask=mask, noise_map=noise_map)
            img = aug['image']
            mask = aug['mask']
            noise_map = aug['noise_map'] # Already tensor (1, 512, 512)
            
        # FIX: Ensure proper dimensions without double unsqueezing
        if mask.dim() == 2: mask = mask.unsqueeze(0)
        # noise_map is likely (1, 512, 512) from ToTensorV2, so we leave it.
        # Just in case it isn't:
        if noise_map.dim() == 2: noise_map = noise_map.unsqueeze(0)
            
        return img, mask, noise_map

# --- 5. EXECUTION ---
def train():
    print("📂 Scanning...")
    auth = glob.glob(os.path.join(CFG.TRAIN_IMG, 'authentic', '*.*'))
    forg = glob.glob(os.path.join(CFG.TRAIN_IMG, 'forged', '*.*'))
    
    if not auth: print("❌ No authentic images!"); return
    
    data = [{'image_path': p, 'label': 0, 'mask_path': None} for p in auth]
    for p in forg:
        base = os.path.splitext(os.path.basename(p))[0]
        mp = os.path.join(CFG.TRAIN_MASK, f"{base}.npy")
        data.append({'image_path': p, 'label': 1, 'mask_path': mp})
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} images.")
    
    tf = A.Compose([A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE), A.Normalize(), ToTensorV2()], additional_targets={'noise_map': 'image'})
    
    train_dl = DataLoader(ForensicDataset(df, tf), batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
    
    model = TitanUltimate().to(CFG.DEVICE)
    opt = optim.AdamW(model.parameters(), lr=CFG.LR)
    
    def criterion(pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        tp = (pred * target).sum()
        fp = ((1-target)*pred).sum()
        fn = (target*(1-pred)).sum()
        tv = (tp + 1e-6) / (tp + CFG.ALPHA*fp + CFG.BETA*fn + 1e-6)
        return (1 - tv)**CFG.GAMMA
    
    print("🚀 Training Started...")
    best_loss = float('inf')
    
    for epoch in range(CFG.EPOCHS):
        model.train()
        loop = tqdm(train_dl, desc=f"Ep {epoch+1}")
        
        for img, mask, noise in loop:
            img, mask, noise = img.to(CFG.DEVICE), mask.to(CFG.DEVICE), noise.to(CFG.DEVICE)
            opt.zero_grad()
            out = model(img, noise)
            loss = criterion(out, mask)
            loss.backward()
            opt.step()
            loop.set_postfix(loss=loss.item())
            
        torch.save(model.state_dict(), CFG.WEIGHTS_NAME)

if __name__ == "__main__":
    train()
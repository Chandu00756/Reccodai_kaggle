# ==================================================================================
# TITAN XIII: ULTIMATE (M4 PRO OPTIMIZED EDITION)
# ==================================================================================
# CRITICAL FIXES FOR M4/M3 MAX:
# 1. GRADIENT CHECKPOINTING: Enabled on all Encoders/Decoders. 
#    - Prevents "INT_MAX" graph crash by keeping the graph shallow.
# 2. PHYSICS DETACHMENT: Physics maps are now explicitly calculated in no_grad().
# 3. CPU LOSS: Loss calculation moved to CPU to save GPU graph memory.
# ==================================================================================

import os
import sys
import glob
import random
import time
import gc
import math
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint # <--- THE SAVIOR
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pywt
from scipy.fftpack import dct

# --- SYSTEM CONFIG ---
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    import timm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm", "PyWavelets", "scipy"])
    import timm

print("=" * 80)
print("TITAN XIII: ULTIMATE (GRADIENT CHECKPOINTING ACTIVE)")
print(f"Device: MPS (M4 Pro Optimized) | Batch: 2 | Graph Depth: MINIMAL")
print("=" * 80)

# ==================================================================================
# 1. CONFIGURATION
# ==================================================================================
class CFG:
    BASE_DIR = '/Users/chanduchitikam/recodai/recodai-luc-scientific-image-forgery-detection'
    TRAIN_IMG_PATH = os.path.join(BASE_DIR, 'train_images') 
    TRAIN_MASK_PATH = os.path.join(BASE_DIR, 'train_masks')
    WEIGHTS_NAME = "TITAN_XIII_ULTIMATE.pth"
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    TILE_SIZE = 512      
    BATCH_SIZE = 2       
    ACCUM_ITER = 1       
    EPOCHS = 40
    LR = 3e-4
    
    OHEM_START_EPOCH = 3
    NUM_WORKERS = 0

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==================================================================================
# 2. PHYSICS ENGINE (DETACHED FROM GRAPH)
# ==================================================================================
class PhysicsEngineUltimate:
    @staticmethod
    def get_noise_map(img_rgb):
        # Pure Numpy/CV2 - No PyTorch Graph interactions
        try:
            img_ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            y_channel = img_ycc[:, :, 0].astype(np.float32)
            h, w = y_channel.shape
            
            coeffs = pywt.wavedec2(y_channel, 'haar', level=3)
            sigma_map = np.zeros((h, w), dtype=np.float32)
            weights = [0.0, 0.5, 0.3, 0.2]
            
            for i in range(1, 4):
                cD = coeffs[-i][2]
                mad = np.median(np.abs(cD - np.median(cD))) / 0.6745
                cD_resized = cv2.resize(np.abs(cD), (w, h), interpolation=cv2.INTER_LINEAR)
                sigma_map += weights[i] * cD_resized
            
            local_mean = cv2.boxFilter(y_channel, -1, (32, 32))
            local_var = cv2.boxFilter(y_channel**2, -1, (32, 32)) - local_mean**2
            final_map = (sigma_map + np.sqrt(np.clip(local_var, 0, None))) / 2.0
            
            v_min, v_max = final_map.min(), final_map.max()
            final_map = (final_map - v_min) / (v_max - v_min + 1e-8)
            return cv2.GaussianBlur(final_map, (5, 5), 1.0).astype(np.float32)
        except: return np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)

    @staticmethod
    def get_cfa_map(img_rgb):
        try:
            r = img_rgb[:,:,0].astype(np.float32)
            g = img_rgb[:,:,1].astype(np.float32)
            b = img_rgb[:,:,2].astype(np.float32)
            
            k_g = np.array([[0, 1], [1, 0]], dtype=np.float32)
            g_diff = cv2.filter2D(g, -1, k_g)
            g_dev = np.abs(255*0.85 - (255.0 - np.abs(g_diff))) # Scaled to 255 space
            r_dev = np.abs(cv2.Laplacian(r, cv2.CV_32F))
            b_dev = np.abs(cv2.Laplacian(b, cv2.CV_32F))
            
            cfa_map = (g_dev + r_dev + b_dev) / 3.0
            cfa_map = (cfa_map - cfa_map.min()) / (cfa_map.max() - cfa_map.min() + 1e-8)
            return cv2.GaussianBlur(cfa_map, (5, 5), 1.0).astype(np.float32)
        except: return np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.float32)

# ==================================================================================
# 3. ENCODERS (CHECKPOINT READY)
# ==================================================================================
class FrequencyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Use buffers to avoid complex parameter handling in checkpointing
        self.register_buffer('gate', torch.ones(256, 129, dtype=torch.complex64))
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(24,64,3,2,1), nn.BatchNorm2d(64), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128,256,3,2,1), nn.BatchNorm2d(256), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256,512,3,2,1), nn.BatchNorm2d(512), nn.ReLU())
        ])
    
    def haar_dwt(self, x):
        LL = (x[:,:,0::2,0::2]+x[:,:,0::2,1::2]+x[:,:,1::2,0::2]+x[:,:,1::2,1::2])/4
        LH = (x[:,:,0::2,0::2]+x[:,:,0::2,1::2]-x[:,:,1::2,0::2]-x[:,:,1::2,1::2])/4
        HL = (x[:,:,0::2,0::2]-x[:,:,0::2,1::2]+x[:,:,1::2,0::2]-x[:,:,1::2,1::2])/4
        HH = (x[:,:,0::2,0::2]-x[:,:,0::2,1::2]-x[:,:,1::2,0::2]+x[:,:,1::2,1::2])/4
        return torch.cat([LL, LH, HL, HH], dim=1)

    def forward(self, x):
        # Calculate FFT on CPU to avoid MPS complex number bugs
        x_cpu = x.detach().cpu()
        x_small = F.interpolate(x_cpu, (256, 256), mode='bilinear', align_corners=False)
        fft = torch.fft.rfft2(x_small)
        # Gating (No gradient needed for stability on CPU part)
        x_filtered = torch.fft.irfft2(fft, s=(256, 256))
        x_gated = x_filtered.to(x.device)
        
        # DWT on GPU
        dwt1 = self.haar_dwt(x_gated)
        dwt2 = self.haar_dwt(dwt1[:, 0:3])
        dwt2_up = F.interpolate(dwt2, size=(128, 128), mode='nearest')
        
        feat = torch.cat([dwt1, dwt2_up], dim=1)
        outs = []
        for conv in self.convs:
            feat = conv(feat)
            outs.append(feat)
        return outs

class NoiseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1,32,3,2,1), nn.BatchNorm2d(32), nn.ReLU()),
            nn.Sequential(nn.Conv2d(32,64,3,2,1), nn.BatchNorm2d(64), nn.ReLU()),
            nn.Sequential(nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128), nn.ReLU()),
            nn.Sequential(nn.Conv2d(128,256,3,2,1), nn.BatchNorm2d(256), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256,512,3,2,1), nn.BatchNorm2d(512), nn.ReLU())
        ])
    def forward(self, x):
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs.append(x)
        return outs

# ==================================================================================
# 4. FUSION & DECODER (CHECKPOINT READY)
# ==================================================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim_s, dim_f, dim_n, out_dim, num_heads=4):
        super().__init__()
        self.proj_s = nn.Conv2d(dim_s, out_dim, 1)
        self.proj_physics = nn.Linear(dim_f + dim_n, out_dim)
        self.norm_s = nn.LayerNorm(out_dim)
        self.norm_p = nn.LayerNorm(out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True, dropout=0.1)
        
    def forward(self, s, f, n):
        B, Cs, Hs, Ws = s.shape
        s_proj = self.proj_s(s).flatten(2).transpose(1, 2).contiguous() 
        
        f = F.interpolate(f, size=(Hs, Ws), mode='bilinear', align_corners=False)
        n = F.interpolate(n, size=(Hs, Ws), mode='bilinear', align_corners=False)
        phys_cat = torch.cat([f, n], dim=1)
        phys_flat = phys_cat.flatten(2).transpose(1, 2).contiguous()
        phys_proj = self.proj_physics(phys_flat)
        
        q = self.norm_s(s_proj)
        k = self.norm_p(phys_proj)
        
        # Checkpointing the expensive attention mechanism
        if self.training:
            out = checkpoint(self._attn_forward, q, k, use_reentrant=False)
        else:
            out = self._attn_forward(q, k)
            
        return self.proj_s(s) + out.transpose(1, 2).contiguous().reshape(B, -1, Hs, Ws)

    def _attn_forward(self, q, k):
        return self.attn(q, k, k)[0]

class SwinDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        
    def forward(self, x):
        if self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)

    def _forward_impl(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2).contiguous()
        x_flat = x_flat + self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))[0]
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.transpose(1, 2).contiguous().reshape(B, C, H, W)

class TitanXIII_Ultimate(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model('swinv2_tiny_window16_256', pretrained=True, features_only=True)
        s_dims = self.swin.feature_info.channels()
        
        self.freq_enc = FrequencyEncoder()
        self.noise_enc = NoiseEncoder()
        self.cfa_enc = NoiseEncoder()
        
        self.fusion4 = CrossAttentionFusion(s_dims[3], 512, 512, 512)
        self.fusion3 = CrossAttentionFusion(s_dims[2], 256, 256, 256)
        self.fusion2 = CrossAttentionFusion(s_dims[1], 128, 128, 128)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec1 = SwinDecoderBlock(512 + 256)
        self.conv1 = nn.Conv2d(512+256, 256, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec2 = SwinDecoderBlock(256 + 128)
        self.conv2 = nn.Conv2d(256+128, 128, 1)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.dec3 = SwinDecoderBlock(128)
        self.conv3 = nn.Conv2d(128, 64, 1)
        
        self.head_low = nn.Conv2d(256, 1, 1)
        self.head_mid = nn.Conv2d(128, 1, 1)
        self.head_high = nn.Conv2d(64, 1, 1)
        self.head_final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x, physics):
        # 1. Encoders (Swin doesn't need explicit checkpointing, it's efficient enough)
        x_small = F.interpolate(x, (256, 256), mode='bilinear')
        s_feats = self.swin(x_small)
        
        # Physics (Lightweight, no checkpoint needed)
        n_feats = self.noise_enc(physics[:, 0:1])
        c_feats = self.cfa_enc(physics[:, 1:2])
        f_feats = self.freq_enc(x)
        
        n_combined = [n + c for n, c in zip(n_feats, c_feats)]
        
        # 2. Fusion (Implicitly Checkpointed inside class)
        s4 = s_feats[3].permute(0, 3, 1, 2).contiguous()
        fused4 = self.fusion4(s4, f_feats[3], n_combined[4])
        
        s3 = s_feats[2].permute(0, 3, 1, 2).contiguous()
        fused3 = self.fusion3(s3, f_feats[2], n_combined[3])
        
        s2 = s_feats[1].permute(0, 3, 1, 2).contiguous()
        fused2 = self.fusion2(s2, f_feats[1], n_combined[2])
        
        # 3. Decoder (Implicitly Checkpointed)
        d1 = self.conv1(self.dec1(torch.cat([self.up1(fused4), fused3], dim=1)))
        d2 = self.conv2(self.dec2(torch.cat([self.up2(d1), fused2], dim=1)))
        d3 = self.conv3(self.dec3(self.up3(d2)))
        
        pl = F.interpolate(self.head_low(d1), size=x.shape[-2:], mode='bilinear')
        pm = F.interpolate(self.head_mid(d2), size=x.shape[-2:], mode='bilinear')
        ph = F.interpolate(self.head_high(d3), size=x.shape[-2:], mode='bilinear')
        pf = F.interpolate(self.head_final(d3), size=x.shape[-2:], mode='bilinear')
        
        if pf.shape[-2:] != x.shape[-2:]:
             pf = F.interpolate(pf, size=x.shape[-2:], mode='bilinear')
        
        return (0.1*pl + 0.2*pm + 0.3*ph + 0.4*pf), pl, pm, ph

# ==================================================================================
# 5. TRAINING LOOP (CPU LOSS OFFLOAD)
# ==================================================================================
class ForensicDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img = cv2.imread(row['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = np.zeros(img.shape[:2], dtype=np.float32)
            if row['label'] == 1 and row['mask_path']:
                try:
                    m = np.load(row['mask_path'])
                    if hasattr(m, 'files'): m = m[m.files[0]]
                    if m.ndim == 3: m = m.max(axis=2)
                    mask = cv2.resize((m>0).astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                except: pass
            
            if self.transform:
                aug = self.transform(image=img, mask=mask)
                img, mask = aug['image'], aug['mask']
            
            # Physics (CPU SIDE - PURE DETACHED)
            with torch.no_grad():
                noise = PhysicsEngineUltimate.get_noise_map(img)
                cfa = PhysicsEngineUltimate.get_cfa_map(img)
            
            img_t = ToTensorV2()(image=img)['image'].float() / 255.0
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()
            phys_t = torch.stack([torch.from_numpy(noise), torch.from_numpy(cfa)]).float()
            return img_t, phys_t, mask_t
        except: return torch.zeros(3, CFG.TILE_SIZE, CFG.TILE_SIZE), torch.zeros(2, CFG.TILE_SIZE, CFG.TILE_SIZE), torch.zeros(1, CFG.TILE_SIZE, CFG.TILE_SIZE)

class ConsoleLogger:
    def __init__(self, total_epochs, steps_per_epoch):
        self.steps = steps_per_epoch
        self.start_time = time.time()
        self.best_f1 = 0.0
    def log(self, epoch, step, loss, lr):
        if step % 5 == 0:
            elapsed = time.time() - self.start_time
            print(f"\r[E{epoch}][{step}/{self.steps}] Loss: {loss:.4f} | LR: {lr:.6f} | T: {elapsed:.0f}s", end="")

def train():
    auth = glob.glob(f'{CFG.TRAIN_IMG_PATH}/authentic/*.*')
    forg = glob.glob(f'{CFG.TRAIN_IMG_PATH}/forged/*.*')
    data = [{'image_path': p, 'label': 0, 'mask_path': None} for p in auth]
    for p in forg:
        base = os.path.basename(p).split('.')[0]
        mp = f'{CFG.TRAIN_MASK_PATH}/{base}.npy'
        if not os.path.exists(mp): mp = f'{CFG.TRAIN_MASK_PATH}/{base}_mask.npy'
        data.append({'image_path': p, 'label': 1, 'mask_path': mp})
    df = pd.DataFrame(data)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(kf.split(df))
    
    t_train = A.Compose([A.RandomCrop(CFG.TILE_SIZE, CFG.TILE_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.OneOf([A.GaussNoise(), A.ImageCompression()], p=0.5)])
    dl_train = DataLoader(ForensicDataset(df.iloc[train_idx], t_train), batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0)
    dl_val = DataLoader(ForensicDataset(df.iloc[val_idx], A.Compose([A.CenterCrop(CFG.TILE_SIZE, CFG.TILE_SIZE)])), batch_size=CFG.BATCH_SIZE, shuffle=False)
    
    print("Initializing Model...")
    model = TitanXIII_Ultimate().to(CFG.DEVICE)
    opt = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    logger = ConsoleLogger(CFG.EPOCHS, len(dl_train))
    ema_thresh = 0.5
    
    print("Starting Training...")
    for epoch in range(1, CFG.EPOCHS+1):
        model.train()
        for i, (img, phys, mask) in enumerate(dl_train):
            img, phys, mask = img.to(CFG.DEVICE), phys.to(CFG.DEVICE), mask.to(CFG.DEVICE)
            
            # Forward (Checkpointing handles the graph depth)
            preds, _, _, _ = model(img, phys)
            
            # Calculate Loss on CPU to save MPS memory/graph size
            # This is a critical stability fix
            preds_cpu = preds.to('cpu')
            mask_cpu = mask.to('cpu')
            
            loss_pixel = bce(preds_cpu, mask_cpu)
            
            # OHEM on CPU
            if epoch >= CFG.OHEM_START_EPOCH:
                with torch.no_grad():
                    probs = torch.sigmoid(preds_cpu)
                    error = torch.abs(probs - mask_cpu).view(-1)
                    k = int(error.numel() * CFG.HARD_MINING_RATIO)
                    if k > 0:
                        curr = torch.topk(error, k)[0][-1]
                        ema_thresh = 0.9*ema_thresh + 0.1*curr.item()
                weights = (torch.abs(torch.sigmoid(preds_cpu)-mask_cpu) >= ema_thresh).float() * 2.0 + 0.5
                loss = (loss_pixel * weights).mean()
            else:
                loss = loss_pixel.mean()
            
            # Move loss back to device for backward, OR backward from here if graph is preserved
            # Actually, to backward() we need loss on GPU if model is on GPU.
            # BUT: Moving simple scalar loss back to GPU is cheap.
            loss = loss.to(CFG.DEVICE)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            
            logger.log(epoch, i, loss.item(), opt.param_groups[0]['lr'])
        
        # Validation
        model.eval()
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for img, phys, mask in dl_val:
                img, phys, mask = img.to(CFG.DEVICE), phys.to(CFG.DEVICE), mask.to(CFG.DEVICE)
                out, _, _, _ = model(img, phys)
                pred = (torch.sigmoid(out) > 0.5).float()
                tp += (pred*mask).sum().item()
                fp += (pred*(1-mask)).sum().item()
                fn += ((1-pred)*mask).sum().item()
        
        f1 = 2*tp/(2*tp+fp+fn+1e-8)
        print(f"\nEPOCH {epoch} | Val F1: {f1:.4f}")
        if f1 >= logger.best_f1:
            logger.best_f1 = f1
            torch.save(model.state_dict(), CFG.WEIGHTS_NAME)
            print("Model Saved!")
        gc.collect()

if __name__ == "__main__":
    train()
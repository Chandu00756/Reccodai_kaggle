"""
TITAN-APEX V2: Forensic Image Forgery Detection
Properly implemented per Apex Architecture specification
"""
import os, gc, cv2, math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pywt, albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
from glob import glob

# ============== CONFIG ==============
class Config:
    BASE = "/Users/chanduchitikam/recodai/recodai-luc-scientific-image-forgery-detection"
    TRAIN_IMG = f"{BASE}/train_images"
    TRAIN_MASK = f"{BASE}/train_masks"
    IMG_SIZE = 384  # Reduced for memory
    BATCH_SIZE = 1
    EPOCHS = 50
    LR = 3e-5  # Lower LR for stability
    DEVICE = "mps"
    NUM_WORKERS = 0
    GRAD_ACCUM = 8  # More accumulation
    ALPHA, BETA, GAMMA = 0.3, 0.7, 1.5

CFG = Config()

# ============== PROPER 30 SRM KERNELS ==============
def get_srm_kernels():
    """30 standard SRM high-pass filters"""
    kernels = []
    
    # 1st order derivatives (4)
    k1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-1,0,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32)
    k2 = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32)
    k3 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,1,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32)
    k4 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,0,0]], np.float32)
    kernels.extend([k1, k2, k3, k4])
    
    # 2nd order (4)
    k5 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32)
    k6 = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]], np.float32)
    k7 = np.array([[0,0,0,0,0],[0,1,0,0,0],[0,0,-2,0,0],[0,0,0,1,0],[0,0,0,0,0]], np.float32)
    k8 = np.array([[0,0,0,0,0],[0,0,0,1,0],[0,0,-2,0,0],[0,1,0,0,0],[0,0,0,0,0]], np.float32)
    kernels.extend([k5, k6, k7, k8])
    
    # 3rd order SPAM (4)
    k9 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,-1,3,-3,1],[0,0,0,0,0],[0,0,0,0,0]], np.float32)
    k10 = np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,0,3,0,0],[0,0,-3,0,0],[0,0,1,0,0]], np.float32)
    k11 = np.array([[0,0,0,0,1],[0,0,0,-3,0],[0,0,3,0,0],[0,-1,0,0,0],[0,0,0,0,0]], np.float32)
    k12 = np.array([[1,0,0,0,0],[0,-3,0,0,0],[0,0,3,0,0],[0,0,0,-1,0],[0,0,0,0,0]], np.float32)
    kernels.extend([k9, k10, k11, k12])
    
    # SQUARE 3x3 (1)
    k13 = np.zeros((5,5), np.float32)
    k13[1:4,1:4] = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32) / 8
    kernels.append(k13)
    
    # SQUARE 5x5 (1)
    k14 = -np.ones((5,5), np.float32) / 24
    k14[2,2] = 1
    kernels.append(k14)
    
    # EDGE 3x3 (4)
    edges = [
        [[-1,2,-1],[2,-4,2],[0,0,0]],
        [[0,2,-1],[0,-4,2],[0,2,-1]],
        [[0,0,0],[2,-4,2],[-1,2,-1]],
        [[-1,2,0],[2,-4,0],[-1,2,0]]
    ]
    for e in edges:
        k = np.zeros((5,5), np.float32)
        k[1:4,1:4] = np.array(e, np.float32) / 4
        kernels.append(k)
    
    # 4th order SQUARE (4) - Min-Max filters
    k19 = np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,-4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]], np.float32) / 4
    k20 = np.array([[-1,0,0,0,-1],[0,2,0,2,0],[0,0,-4,0,0],[0,2,0,2,0],[-1,0,0,0,-1]], np.float32) / 4
    k21 = np.array([[0,-1,0,-1,0],[-1,2,2,2,-1],[0,2,-8,2,0],[-1,2,2,2,-1],[0,-1,0,-1,0]], np.float32) / 8
    k22 = np.array([[1,-2,1,0,0],[-2,4,-2,0,0],[1,-2,1,0,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32) / 4
    kernels.extend([k19, k20, k21, k22])
    
    # Sobel-like (4)
    k23 = np.array([[0,0,0,0,0],[0,-1,-2,-1,0],[0,0,0,0,0],[0,1,2,1,0],[0,0,0,0,0]], np.float32) / 4
    k24 = np.array([[0,0,0,0,0],[0,-1,0,1,0],[0,-2,0,2,0],[0,-1,0,1,0],[0,0,0,0,0]], np.float32) / 4
    k25 = np.array([[0,0,0,0,0],[0,0,0,0,0],[1,2,0,-2,-1],[0,0,0,0,0],[0,0,0,0,0]], np.float32) / 4
    k26 = np.array([[0,0,1,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,-2,0,0],[0,0,-1,0,0]], np.float32) / 4
    kernels.extend([k23, k24, k25, k26])
    
    # Prewitt-like (4)
    k27 = np.array([[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0],[0,-1,-1,-1,0],[0,0,0,0,0]], np.float32) / 3
    k28 = np.array([[0,0,0,0,0],[0,-1,0,1,0],[0,-1,0,1,0],[0,-1,0,1,0],[0,0,0,0,0]], np.float32) / 3
    k29 = np.array([[0,0,0,0,0],[0,1,0,-1,0],[0,1,0,-1,0],[0,1,0,-1,0],[0,0,0,0,0]], np.float32) / 3
    k30 = np.array([[0,0,0,0,0],[0,-1,-1,-1,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0]], np.float32) / 3
    kernels.extend([k27, k28, k29, k30])
    
    return np.stack(kernels[:30], axis=0)

# ============== SRM LAYER ==============
class SRMConv2d(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        srm = get_srm_kernels()  # (30, 5, 5)
        srm_rgb = np.stack([srm]*in_ch, axis=1)  # (30, 3, 5, 5)
        self.register_buffer('weight', torch.from_numpy(srm_rgb))
        
    def forward(self, x):
        out = F.conv2d(x, self.weight, padding=2)
        return torch.clamp(out, -3.0, 3.0)  # TLU

# ============== BAYAR LAYER ==============
class BayarConv2d(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, 5, 5) * 0.02)
        
    def forward(self, x):
        w = self.weight.clone()
        # Mask out center, normalize, set center to -1
        mask = torch.ones_like(w)
        mask[:, :, 2, 2] = 0
        w = w * mask
        w_sum = w.abs().sum(dim=(2,3), keepdim=True).clamp(min=1e-8)
        w = w / w_sum
        w[:, :, 2, 2] = -1
        return F.conv2d(x, w, padding=2)

# ============== PHYSICS MAPS ==============
def compute_physics_maps(img_tensor):
    """Compute noise residual, local variance, gradient magnitude - returns 4 channels"""
    B, C, H, W = img_tensor.shape
    device = img_tensor.device
    
    # Convert to grayscale
    gray = 0.299 * img_tensor[:,0] + 0.587 * img_tensor[:,1] + 0.114 * img_tensor[:,2]
    gray = gray.unsqueeze(1)  # (B, 1, H, W)
    
    # Gradient magnitude (Sobel)
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=torch.float32).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=torch.float32).view(1,1,3,3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    
    # Local variance (3x3 window)
    mean_filter = torch.ones(1, 1, 3, 3, device=device) / 9
    local_mean = F.conv2d(gray, mean_filter, padding=1)
    local_var = F.conv2d((gray - local_mean)**2, mean_filter, padding=1)
    
    # High-pass residual (Laplacian)
    laplacian = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], device=device, dtype=torch.float32).view(1,1,3,3)
    noise_res = F.conv2d(gray, laplacian, padding=1).abs()
    
    # Extra channel - edge response
    edge = (gx.abs() + gy.abs()) / 2
    
    return torch.cat([grad_mag, local_var, noise_res, edge], dim=1)  # (B, 4, H, W)

# ============== ENCODER ==============
class DualStreamEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Semantic stream (EfficientNet for speed)
        # out_indices=(0,1,2,3) gives channels: 24, 32, 48, 136
        self.semantic = timm.create_model('efficientnet_b3', pretrained=True, features_only=True, out_indices=(0,1,2,3))
        self.sem_channels = [24, 32, 48, 136]  # EfficientNet-B3 feature channels
        
        # Forensic stream
        self.srm = SRMConv2d(3)
        self.bayar = BayarConv2d(3, 3)
        
        # Process SRM (30 ch) + Bayar (3 ch) + Physics (4 ch) = 37 ch
        self.forensic_stem = nn.Sequential(
            nn.Conv2d(37, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        
        # Forensic encoder stages matching semantic
        self.f_stage1 = self._make_stage(64, 24, stride=1)
        self.f_stage2 = self._make_stage(24, 32, stride=2)
        self.f_stage3 = self._make_stage(32, 48, stride=2)
        self.f_stage4 = self._make_stage(48, 136, stride=2)
        
    def _make_stage(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Semantic features
        sem_feats = self.semantic(x)  # List of 4 feature maps
        
        # Forensic features
        srm_out = self.srm(x)  # (B, 30, H, W)
        bayar_out = self.bayar(x)  # (B, 3, H, W)
        physics = compute_physics_maps(x)  # (B, 4, H, W)
        
        forensic_in = torch.cat([srm_out, bayar_out, physics], dim=1)  # (B, 37, H, W)
        f0 = self.forensic_stem(forensic_in)  # (B, 64, H/2, W/2)
        
        f1 = self.f_stage1(f0)
        f2 = self.f_stage2(f1)
        f3 = self.f_stage3(f2)
        f4 = self.f_stage4(f3)
        
        return sem_feats, [f1, f2, f3, f4]

# ============== CROSS ATTENTION ==============
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Conv2d(dim, dim//4, 1)
        self.k = nn.Conv2d(dim, dim//4, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, sem, foren):
        B, C, H, W = sem.shape
        # Match sizes
        if foren.shape[2:] != sem.shape[2:]:
            foren = F.interpolate(foren, size=(H, W), mode='bilinear', align_corners=False)
        
        q = self.q(sem).reshape(B, -1, H*W).permute(0, 2, 1)  # (B, HW, C/4)
        k = self.k(foren).reshape(B, -1, H*W)  # (B, C/4, HW)
        v = self.v(foren).reshape(B, C, H*W)  # (B, C, HW)
        
        attn = torch.bmm(q, k) / math.sqrt(C//4)  # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(B, C, H, W)
        return sem + self.gamma * out

# ============== DECODER ==============
class Decoder(nn.Module):
    def __init__(self, channels=[24, 32, 48, 136]):
        super().__init__()
        self.attn4 = CrossAttention(channels[3])
        self.attn3 = CrossAttention(channels[2])
        self.attn2 = CrossAttention(channels[1])
        self.attn1 = CrossAttention(channels[0])
        
        self.up4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[2], 3, padding=1),
            nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(channels[2]*2, channels[1], 3, padding=1),
            nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[1]*2, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(channels[0]*2, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.head = nn.Conv2d(32, 1, 1)
        
    def forward(self, sem_feats, for_feats, target_size):
        s1, s2, s3, s4 = sem_feats
        f1, f2, f3, f4 = for_feats
        
        # Fuse at each level
        x = self.attn4(s4, f4)
        x = self.up4(x)
        
        x = torch.cat([x, self.attn3(s3, f3)], dim=1)
        x = self.up3(x)
        
        x = torch.cat([x, self.attn2(s2, f2)], dim=1)
        x = self.up2(x)
        
        x = torch.cat([x, self.attn1(s1, f1)], dim=1)
        x = self.up1(x)
        
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.head(x)

# ============== TITAN APEX MODEL ==============
class TitanApex(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DualStreamEncoder()
        self.decoder = Decoder([24, 32, 48, 136])
        
    def forward(self, x):
        sem_feats, for_feats = self.encoder(x)
        return self.decoder(sem_feats, for_feats, (x.shape[2], x.shape[3]))

# ============== LOSS ==============
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        # BCE component for stability
        bce_loss = self.bce(pred, target)
        
        # Focal Tversky
        pred_sig = torch.sigmoid(pred)
        pred_flat = pred_sig.view(-1)
        target_flat = target.view(-1)
        
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (tp + 1) / (tp + self.alpha * fp + self.beta * fn + 1)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return 0.5 * bce_loss + 0.5 * focal_tversky

# ============== DATASET ==============
class ForensicDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.mask_dir = mask_dir
        
        self.forged = glob(f"{img_dir}/forged/*.png") + glob(f"{img_dir}/forged/*.jpg")
        self.authentic = glob(f"{img_dir}/authentic/*.png") + glob(f"{img_dir}/authentic/*.jpg")
        self.samples = [(p, True) for p in self.forged] + [(p, False) for p in self.authentic]
        
        print(f"Dataset: {len(self.forged)} forged + {len(self.authentic)} authentic = {len(self.samples)} total")
        
        if is_train:
            self.aug = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.aug = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, is_forged = self.samples[idx]
        
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        if is_forged:
            img_id = os.path.splitext(os.path.basename(path))[0]
            mask_path = os.path.join(self.mask_dir, f"{img_id}.npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path).astype(np.float32)
                if mask.ndim > 2:
                    mask = mask.squeeze()
                if mask.size > 0 and mask.ndim == 2 and mask.shape[0] > 0 and mask.shape[1] > 0:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.ones((h, w), np.float32)
            else:
                mask = np.ones((h, w), np.float32)
        else:
            mask = np.zeros((h, w), np.float32)
        
        augmented = self.aug(image=img, mask=mask)
        return augmented['image'], augmented['mask'].unsqueeze(0)

# ============== TRAINING ==============
def train():
    print("=" * 50)
    print("TITAN-APEX V2 Training")
    print("=" * 50)
    
    dataset = ForensicDataset(CFG.TRAIN_IMG, CFG.TRAIN_MASK, CFG.IMG_SIZE, is_train=True)
    n_val = len(dataset) // 5
    train_ds, val_ds = torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    model = TitanApex().to(CFG.DEVICE)
    
    # Warmup + Cosine schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=CFG.LR * 10, epochs=CFG.EPOCHS,
        steps_per_epoch=len(train_loader) // CFG.GRAD_ACCUM,
        pct_start=0.1  # 10% warmup
    )
    
    criterion = CombinedLoss(CFG.ALPHA, CFG.BETA, CFG.GAMMA)
    best_f1 = 0
    
    for epoch in range(CFG.EPOCHS):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}")
        for i, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(CFG.DEVICE)
            masks = masks.to(CFG.DEVICE)
            
            preds = model(imgs)
            loss = criterion(preds, masks) / CFG.GRAD_ACCUM
            loss.backward()
            
            if (i + 1) % CFG.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += loss.item() * CFG.GRAD_ACCUM
            pbar.set_postfix({'loss': f'{train_loss/(i+1):.4f}'})
            
            del imgs, masks, preds, loss
            if i % 20 == 0:
                gc.collect()
        
        # Proper F1 calculation (accumulate TP/FP/FN)
        model.eval()
        total_tp, total_fp, total_fn = 0, 0, 0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(CFG.DEVICE)
                masks = masks.to(CFG.DEVICE)
                
                preds = (torch.sigmoid(model(imgs)) > 0.5).float()
                
                total_tp += (preds * masks).sum().item()
                total_fp += (preds * (1 - masks)).sum().item()
                total_fn += ((1 - preds) * masks).sum().item()
                
                del imgs, masks, preds
        
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        val_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}, P={precision:.4f}, R={recall:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "TITAN_APEX_BEST.pth")
            print(f"Saved best model with F1={best_f1:.4f}")
        
        gc.collect()
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()

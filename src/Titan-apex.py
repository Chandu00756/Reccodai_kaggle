"""
TITAN-APEX V3: Maximum Speed + 0.8 F1 Target
Optimized for M4 Pro 32GB RAM
"""
import os, gc, cv2, math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import timm
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ============== OPTIMIZED CONFIG ==============
class Config:
    BASE = "/Users/chanduchitikam/recodai/recodai-luc-scientific-image-forgery-detection"
    TRAIN_IMG = f"{BASE}/train_images"
    TRAIN_MASK = f"{BASE}/train_masks"
    
    # M4 Pro optimized settings
    IMG_SIZE = 384
    BATCH_SIZE = 8  # Increased for M4 Pro 32GB
    EPOCHS = 40
    LR = 1e-4
    GRAD_ACCUM = 2  # Effective batch = 16
    NUM_WORKERS = 6  # M4 Pro has 12 cores
    
    # Training optimizations
    WARMUP_EPOCHS = 2
    WEIGHT_DECAY = 1e-4
    
    # Paths
    SAVE_PATH = "/Users/chanduchitikam/recodai/TITAN_APEX_BEST.pth"

# ============== MPS OPTIMIZATIONS ==============
def get_device():
    """Get device without printing (for worker processes)"""
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()

# ============== PRECOMPUTED SRM KERNELS (30 filters) ==============
def get_srm_kernels():
    """30 SRM high-pass filters - precomputed once"""
    kernels = []
    
    # 1st order (4)
    kernels.append(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-1,0,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32))
    kernels.append(np.array([[0,0,0,0,0],[0,0,1,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32))
    kernels.append(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,1,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32))
    kernels.append(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,-1,0,0],[0,0,1,0,0],[0,0,0,0,0]], np.float32))
    
    # 2nd order (4)
    kernels.append(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32))
    kernels.append(np.array([[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]], np.float32))
    kernels.append(np.array([[0,0,0,0,0],[0,1,0,0,0],[0,0,-2,0,0],[0,0,0,1,0],[0,0,0,0,0]], np.float32))
    kernels.append(np.array([[0,0,0,0,0],[0,0,0,1,0],[0,0,-2,0,0],[0,1,0,0,0],[0,0,0,0,0]], np.float32))
    
    # 3rd order SPAM (4)
    kernels.append(np.array([[0,0,0,0,0],[0,0,0,0,0],[0,-1,3,-3,1],[0,0,0,0,0],[0,0,0,0,0]], np.float32))
    kernels.append(np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,0,3,0,0],[0,0,-3,0,0],[0,0,1,0,0]], np.float32))
    kernels.append(np.array([[0,0,0,0,1],[0,0,0,-3,0],[0,0,3,0,0],[0,-1,0,0,0],[0,0,0,0,0]], np.float32))
    kernels.append(np.array([[1,0,0,0,0],[0,-3,0,0,0],[0,0,3,0,0],[0,0,0,-1,0],[0,0,0,0,0]], np.float32))
    
    # SQUARE 3x3 (1)
    k = np.zeros((5,5), np.float32)
    k[1:4,1:4] = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32) / 8
    kernels.append(k)
    
    # SQUARE 5x5 (1)
    k = -np.ones((5,5), np.float32) / 24
    k[2,2] = 1
    kernels.append(k)
    
    # EDGE 3x3 (4)
    edges = [[[-1,2,-1],[2,-4,2],[0,0,0]],[[0,2,-1],[0,-4,2],[0,2,-1]],
             [[0,0,0],[2,-4,2],[-1,2,-1]],[[-1,2,0],[2,-4,0],[-1,2,0]]]
    for e in edges:
        k = np.zeros((5,5), np.float32)
        k[1:4,1:4] = np.array(e, np.float32) / 4
        kernels.append(k)
    
    # 4th order (4)
    kernels.append(np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,-4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]], np.float32)/4)
    kernels.append(np.array([[-1,0,0,0,-1],[0,2,0,2,0],[0,0,-4,0,0],[0,2,0,2,0],[-1,0,0,0,-1]], np.float32)/4)
    kernels.append(np.array([[0,-1,0,-1,0],[-1,2,2,2,-1],[0,2,-8,2,0],[-1,2,2,2,-1],[0,-1,0,-1,0]], np.float32)/8)
    kernels.append(np.array([[1,-2,1,0,0],[-2,4,-2,0,0],[1,-2,1,0,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32)/4)
    
    # Sobel-like (4)
    kernels.append(np.array([[0,0,0,0,0],[0,-1,-2,-1,0],[0,0,0,0,0],[0,1,2,1,0],[0,0,0,0,0]], np.float32)/4)
    kernels.append(np.array([[0,0,0,0,0],[0,-1,0,1,0],[0,-2,0,2,0],[0,-1,0,1,0],[0,0,0,0,0]], np.float32)/4)
    kernels.append(np.array([[0,0,0,0,0],[0,0,0,0,0],[1,2,0,-2,-1],[0,0,0,0,0],[0,0,0,0,0]], np.float32)/4)
    kernels.append(np.array([[0,0,1,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,-2,0,0],[0,0,-1,0,0]], np.float32)/4)
    
    # Prewitt-like (4)
    kernels.append(np.array([[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0],[0,-1,-1,-1,0],[0,0,0,0,0]], np.float32)/3)
    kernels.append(np.array([[0,0,0,0,0],[0,-1,0,1,0],[0,-1,0,1,0],[0,-1,0,1,0],[0,0,0,0,0]], np.float32)/3)
    kernels.append(np.array([[0,0,0,0,0],[0,1,0,-1,0],[0,1,0,-1,0],[0,1,0,-1,0],[0,0,0,0,0]], np.float32)/3)
    kernels.append(np.array([[0,0,0,0,0],[0,-1,-1,-1,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0]], np.float32)/3)
    
    return np.stack(kernels[:30], axis=0)

# Global precomputed kernels
SRM_KERNELS = get_srm_kernels()

# ============== FAST SRM LAYER ==============
class SRMConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        kernels_rgb = np.tile(SRM_KERNELS[:, np.newaxis, :, :], (1, 3, 1, 1)) / 3
        self.register_buffer('weight', torch.from_numpy(kernels_rgb).float())
        
    def forward(self, x):
        return torch.clamp(F.conv2d(x, self.weight, padding=2), -3, 3)

# ============== BAYAR CONSTRAINED CONV ==============
class BayarConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3, 5, 5) * 0.01)
        
    def forward(self, x):
        w = self.weight.clone()
        w[:, :, 2, 2] = 0
        w = w - w.sum(dim=(2,3), keepdim=True) / 24
        w[:, :, 2, 2] = -1
        return F.conv2d(x, w, padding=2)

# ============== FAST PHYSICS MAPS ==============
class PhysicsMaps(nn.Module):
    """Pre-registered buffers for speed"""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        self.register_buffer('sobel_x', torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('sobel_y', torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('mean_k', torch.ones(1,1,3,3, dtype=torch.float32) / 9)
        self.register_buffer('laplacian', torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3))
        
    def forward(self, x):
        B = x.shape[0]
        gray = (0.299 * x[:,0] + 0.587 * x[:,1] + 0.114 * x[:,2]).unsqueeze(1)
        
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
        
        local_mean = F.conv2d(gray, self.mean_k, padding=1)
        local_var = F.conv2d((gray - local_mean)**2, self.mean_k, padding=1)
        
        noise_res = F.conv2d(gray, self.laplacian, padding=1).abs()
        edge = (gx.abs() + gy.abs()) / 2
        
        return torch.cat([grad_mag, local_var, noise_res, edge], dim=1)

# ============== LIGHTWEIGHT ATTENTION ==============
class EfficientAttention(nn.Module):
    """Fast attention with reduced computation"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = (dim // 8) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 2, 1, bias=False)  # Only Q and KV combined
        self.proj = nn.Conv2d(dim, dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, sem, foren):
        B, C, H, W = sem.shape
        if foren.shape[2:] != sem.shape[2:]:
            foren = F.interpolate(foren, size=(H, W), mode='bilinear', align_corners=False)
        
        # Simple additive fusion with learned weight
        fused = sem + self.gamma * self.proj(foren)
        return fused

# ============== ENCODER ==============
class DualStreamEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Semantic - EfficientNet-B3 (fast + accurate)
        self.semantic = timm.create_model('efficientnet_b3', pretrained=True, 
                                          features_only=True, out_indices=(0,1,2,3))
        self.sem_channels = [24, 32, 48, 136]
        
        # Forensic stream
        self.srm = SRMConv2d()
        self.bayar = BayarConv2d()
        self.physics = PhysicsMaps()
        
        # 30 SRM + 3 Bayar + 4 Physics = 37 channels
        self.forensic_stem = nn.Sequential(
            nn.Conv2d(37, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        
        # Forensic stages to match semantic
        self.f1 = nn.Sequential(nn.Conv2d(64, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True))
        self.f2 = nn.Sequential(nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.f3 = nn.Sequential(nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.f4 = nn.Sequential(nn.Conv2d(48, 136, 3, stride=2, padding=1), nn.BatchNorm2d(136), nn.ReLU(inplace=True))
        
    def forward(self, x):
        # Semantic features
        sem_feats = self.semantic(x)
        
        # Forensic features
        srm_out = self.srm(x)
        bayar_out = self.bayar(x)
        physics_out = self.physics(x)
        
        forensic_in = torch.cat([srm_out, bayar_out, physics_out], dim=1)
        f0 = self.forensic_stem(forensic_in)
        
        f1 = self.f1(f0)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        f4 = self.f4(f3)
        
        return sem_feats, [f1, f2, f3, f4]

# ============== FAST DECODER ==============
class Decoder(nn.Module):
    def __init__(self, channels=[24, 32, 48, 136]):
        super().__init__()
        self.attn4 = EfficientAttention(channels[3])
        self.attn3 = EfficientAttention(channels[2])
        self.attn2 = EfficientAttention(channels[1])
        self.attn1 = EfficientAttention(channels[0])
        
        # Decoder blocks
        self.up4 = nn.Sequential(
            nn.Conv2d(channels[3], channels[2], 3, padding=1), nn.BatchNorm2d(channels[2]), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(
            nn.Conv2d(channels[2]*2, channels[1], 3, padding=1), nn.BatchNorm2d(channels[1]), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[1]*2, channels[0], 3, padding=1), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(
            nn.Conv2d(channels[0]*2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        
        # Initialize output bias positive to encourage predictions > 0.5
        self.head[-1].bias.data.fill_(0.5)
        
    def forward(self, sem_feats, for_feats, target_size):
        s1, s2, s3, s4 = sem_feats
        f1, f2, f3, f4 = for_feats
        
        # Level 4
        x = self.attn4(s4, f4)
        x = self.up4(x)
        x = F.interpolate(x, size=s3.shape[2:], mode='bilinear', align_corners=False)
        
        # Level 3
        x = torch.cat([x, self.attn3(s3, f3)], dim=1)
        x = self.up3(x)
        x = F.interpolate(x, size=s2.shape[2:], mode='bilinear', align_corners=False)
        
        # Level 2
        x = torch.cat([x, self.attn2(s2, f2)], dim=1)
        x = self.up2(x)
        x = F.interpolate(x, size=s1.shape[2:], mode='bilinear', align_corners=False)
        
        # Level 1
        x = torch.cat([x, self.attn1(s1, f1)], dim=1)
        x = self.up1(x)
        
        # Final upsampling to target size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.head(x)

# ============== MAIN MODEL ==============
class TitanApex(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DualStreamEncoder()
        self.decoder = Decoder()
        
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
        bce_loss = self.bce(pred, target)
        
        pred_sig = torch.sigmoid(pred)
        tp = (pred_sig * target).sum(dim=(2,3))
        fp = (pred_sig * (1-target)).sum(dim=(2,3))
        fn = ((1-pred_sig) * target).sum(dim=(2,3))
        
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        focal_tversky = ((1 - tversky) ** self.gamma).mean()
        
        return 0.4 * bce_loss + 0.6 * focal_tversky

# ============== FAST DATASET ==============
class ForensicDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train
        self.size = Config.IMG_SIZE
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path, is_forged = self.samples[idx]
        
        # Fast image loading
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        
        # Load mask - handle various formats
        mask = np.zeros((self.size, self.size), dtype=np.float32)
        if mask_path and os.path.exists(mask_path):
            try:
                raw_mask = np.load(mask_path)
                # Handle 3D masks (take first channel or max across channels)
                if len(raw_mask.shape) == 3:
                    raw_mask = raw_mask.max(axis=-1) if raw_mask.shape[-1] <= 4 else raw_mask[:,:,0]
                if raw_mask.size > 0 and raw_mask.shape[0] > 0 and raw_mask.shape[1] > 0:
                    mask = cv2.resize(raw_mask.astype(np.float32), (self.size, self.size), 
                                     interpolation=cv2.INTER_LINEAR)
                    mask = (mask > 0.5).astype(np.float32)
            except Exception:
                pass
        
        # Training augmentations
        if self.is_train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1, :].copy()
                mask = mask[::-1, :].copy()
            if np.random.rand() < 0.3:
                k = np.random.randint(1, 4)
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
            # Color jitter
            if np.random.rand() < 0.3:
                img = img.astype(np.float32)
                img *= np.random.uniform(0.8, 1.2)
                img = np.clip(img, 0, 255).astype(np.uint8)
            # JPEG compression artifact (critical for generalization)
            if np.random.rand() < 0.4:
                quality = np.random.randint(60, 95)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encimg = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
                img = cv2.cvtColor(cv2.imdecode(encimg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # Gaussian noise
            if np.random.rand() < 0.2:
                noise = np.random.normal(0, np.random.uniform(5, 15), img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return img, mask

# ============== TRAINING ==============
def train():
    print("=" * 60)
    print("TITAN-APEX V3: M4 Pro Optimized Training")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Gather samples
    forged_imgs = sorted(glob(f"{Config.TRAIN_IMG}/forged/*.png"))
    auth_imgs = sorted(glob(f"{Config.TRAIN_IMG}/authentic/*.png"))
    
    samples = []
    for p in forged_imgs:
        img_id = os.path.basename(p).replace('.png', '')
        mask_path = f"{Config.TRAIN_MASK}/{img_id}.npy"
        samples.append((p, mask_path, True))
    for p in auth_imgs:
        samples.append((p, None, False))
    
    print(f"Dataset: {len(forged_imgs)} forged + {len(auth_imgs)} authentic = {len(samples)} total")
    
    # Split with stratification
    np.random.seed(42)
    np.random.shuffle(samples)
    forged_samples = [s for s in samples if s[2]]
    auth_samples = [s for s in samples if not s[2]]
    
    n_val_forged = int(len(forged_samples) * 0.2)
    n_val_auth = int(len(auth_samples) * 0.2)
    
    val_samples = forged_samples[:n_val_forged] + auth_samples[:n_val_auth]
    train_samples = forged_samples[n_val_forged:] + auth_samples[n_val_auth:]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Config: IMG_SIZE={Config.IMG_SIZE}, BATCH={Config.BATCH_SIZE}, ACCUM={Config.GRAD_ACCUM}")
    print(f"Effective batch size: {Config.BATCH_SIZE * Config.GRAD_ACCUM}")
    
    # Dataloaders - optimized for M4 Pro
    train_ds = ForensicDataset(train_samples, is_train=True)
    val_ds = ForensicDataset(val_samples, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=False,
                              drop_last=False, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE * 2, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=False,
                            persistent_workers=True)
    
    # Model
    model = TitanApex().to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss & Optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # Scheduler - use ceil to account for partial batches
    steps_per_epoch = math.ceil(len(train_loader) / Config.GRAD_ACCUM)
    total_steps = steps_per_epoch * Config.EPOCHS
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LR * 10,
        total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    best_f1 = 0
    
    for epoch in range(Config.EPOCHS):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        accum_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for i, (imgs, masks) in enumerate(pbar):
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            preds = model(imgs)
            loss = criterion(preds, masks) / Config.GRAD_ACCUM
            loss.backward()
            accum_count += 1
            
            if accum_count >= Config.GRAD_ACCUM or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
            
            train_loss += loss.item() * Config.GRAD_ACCUM
            pbar.set_postfix({'loss': f'{loss.item() * Config.GRAD_ACCUM:.4f}', 
                             'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation - compute proper F1
        model.eval()
        val_loss = 0
        total_tp, total_fp, total_fn = 0, 0, 0
        all_preds, all_masks = [], []
        
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="Validating", leave=False):
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()
                
                # Store for analysis
                pred_probs = torch.sigmoid(preds)
                all_preds.append(pred_probs.cpu())
                all_masks.append(masks.cpu())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Analyze predictions and find optimal threshold
        all_preds = torch.cat(all_preds)
        all_masks = torch.cat(all_masks)
        pred_min, pred_max, pred_mean = all_preds.min().item(), all_preds.max().item(), all_preds.mean().item()
        mask_sum = all_masks.sum().item()
        
        # Try multiple thresholds to find best F1
        best_thresh_f1, best_thresh = 0, 0.5
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pred_binary = (all_preds > thresh).float()
            tp = (pred_binary * all_masks).sum().item()
            fp = (pred_binary * (1 - all_masks)).sum().item()
            fn = ((1 - pred_binary) * all_masks).sum().item()
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            f1_t = 2 * prec * rec / (prec + rec + 1e-7)
            if f1_t > best_thresh_f1:
                best_thresh_f1, best_thresh = f1_t, thresh
                total_tp, total_fp, total_fn = tp, fp, fn
        
        # Compute F1 with best threshold
        precision = total_tp / (total_tp + total_fp + 1e-7)
        recall = total_tp / (total_tp + total_fn + 1e-7)
        f1 = best_thresh_f1
        
        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        print(f"  Pred stats: min={pred_min:.4f}, max={pred_max:.4f}, mean={pred_mean:.4f}")
        print(f"  Best thresh={best_thresh:.2f}, TP={total_tp:.0f}, FP={total_fp:.0f}, FN={total_fn:.0f}")
        print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Save best
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
            }, Config.SAVE_PATH)
            print(f"  ★ New best F1: {f1:.4f} - Saved!")
        
        # Early progress check
        if epoch == 4 and f1 < 0.1:
            print("\n⚠️ Warning: F1 very low after 5 epochs. Consider adjusting hyperparameters.")
        
        gc.collect()
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {Config.SAVE_PATH}")
    print(f"{'='*60}")
    
    return best_f1

if __name__ == "__main__":
    train()

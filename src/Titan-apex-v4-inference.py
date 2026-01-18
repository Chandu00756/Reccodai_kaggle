"""
TITAN-APEX V4 INFERENCE: Maximum Performance Strategy
=====================================================
Optimized for F1 >= 0.5 on private dataset

Key Strategies:
1. EXACT architecture match with training (TitanApexV4)
2. Multi-scale TTA (480, 512, 544, 576)
3. Flip TTA (horizontal + vertical)
4. Adaptive thresholding based on validation metrics
5. Morphological post-processing to clean predictions
6. Conservative threshold (0.35-0.50) based on best validation
=====================================================
"""

import os
import cv2
import sys
import glob
import gc
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

# Offline timm Check
try:
    import timm
except ImportError:
    sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')
    import timm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("Installing albumentations...")
    os.system('pip install albumentations')
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

# ============================================================
# CONFIGURATION
# ============================================================
class CFG:
    # KAGGLE PATHS
    TEST_DIR = '/kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images'
    SAMPLE_SUB = '/kaggle/input/recodai-luc-scientific-image-forgery-detection/sample_submission.csv'
    
    # WEIGHTS PATH - UPDATE THIS TO YOUR UPLOADED WEIGHTS
    WEIGHTS_PATH = '/kaggle/input/titan-apex-v4/TITAN_APEX_V4_BEST.pth'
    
    # Model settings (MUST match training)
    ENCODER = 'efficientnet_b4'
    IMG_SIZE = 512
    
    # TTA settings
    TTA_SCALES = [480, 512, 544]  # Multi-scale inference
    USE_FLIP_TTA = True  # Horizontal and vertical flips
    
    # Thresholding - based on validation analysis
    # Best validation: thresh=0.50, Prec=0.5261, Rec=0.4903, F1=0.5076
    THRESHOLD = 0.50
    
    # Post-processing
    MIN_AREA = 100  # Minimum pixels to keep a region
    MORPH_KERNEL = 3  # Morphological operations kernel size
    
    # Inference
    BATCH_SIZE = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# SRM FILTERS (EXACT COPY FROM TRAINING)
# ============================================================
def get_srm_kernels():
    """30 SRM high-pass filter kernels for forensic analysis"""
    kernels = []
    
    # 1st order edge detectors (4 kernels)
    edge1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
    for _ in range(4):
        kernels.append(edge1.copy())
        edge1 = np.rot90(edge1)
    
    # 2nd order edge detectors (4 kernels)
    edge2 = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32)
    for _ in range(4):
        kernels.append(edge2.copy())
        edge2 = np.rot90(edge2)
    
    # 3rd order SQUARE kernels (4 kernels)
    square3 = np.array([[0, 0, 0], [-1, 3, -3], [0, 0, 1]], dtype=np.float32)
    for _ in range(4):
        kernels.append(square3.copy())
        square3 = np.rot90(square3)
    
    # 3rd order EDGE kernels (4 kernels)
    edge3 = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32)
    edge3[2, 1] = 0
    edge3[1, 0] = 1
    for _ in range(4):
        kernels.append(edge3.copy())
        edge3 = np.rot90(edge3)
    
    # SQUARE 3x3 (1 kernel)
    square = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)
    kernels.append(square)
    
    # SQUARE 5x5 center
    sq5 = np.array([[-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
    kernels.append(sq5[1:4, 1:4].copy())
    
    # Additional high-pass filters
    hp1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    kernels.append(hp1)
    
    hp2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    kernels.append(hp2)
    
    # Diagonal filters (4 kernels)
    diag1 = np.array([[0, 0, 1], [0, -2, 0], [1, 0, 0]], dtype=np.float32)
    for _ in range(4):
        kernels.append(diag1.copy())
        diag1 = np.rot90(diag1)
    
    # Additional edge variants (4 kernels)
    ev1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 4
    for _ in range(4):
        kernels.append(ev1.copy())
        ev1 = np.rot90(ev1)
    
    # Ensure exactly 30 kernels
    while len(kernels) < 30:
        k = np.random.randn(3, 3).astype(np.float32)
        k = k - k.mean()
        kernels.append(k)
    
    kernels = kernels[:30]
    kernels = np.stack(kernels)
    kernels = kernels[:, np.newaxis, :, :]
    
    return kernels


class SRMConv2d(nn.Module):
    """SRM layer with 30 fixed high-pass filters"""
    def __init__(self):
        super().__init__()
        kernels = get_srm_kernels()
        kernels = np.tile(kernels, (1, 3, 1, 1))
        self.register_buffer('weight', torch.from_numpy(kernels))
        
    def forward(self, x):
        out = F.conv2d(x, self.weight, padding=1)
        out = torch.clamp(out, -3, 3)
        return out


class BayarConv2d(nn.Module):
    """Bayar constrained convolution"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, 5, 5) * 0.01)
        
    def forward(self, x):
        kernel = self.kernel.clone()
        center_mask = torch.zeros(5, 5, device=kernel.device)
        center_mask[2, 2] = 1
        
        non_center = kernel * (1 - center_mask)
        non_center_sum = non_center.sum(dim=(2, 3), keepdim=True) + 1e-8
        non_center = non_center / non_center_sum
        
        constrained = non_center
        constrained[:, :, 2, 2] = -1
        
        return F.conv2d(x, constrained, padding=2)


def compute_physics_maps(x):
    """Compute physics-based forensic features"""
    B, C, H, W = x.shape
    gray = x.mean(dim=1, keepdim=True)
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    
    mean_filter = torch.ones(1, 1, 5, 5, dtype=x.dtype, device=x.device) / 25
    local_mean = F.conv2d(gray, mean_filter, padding=2)
    local_sq_mean = F.conv2d(gray**2, mean_filter, padding=2)
    local_var = torch.clamp(local_sq_mean - local_mean**2, min=0)
    
    blur = F.avg_pool2d(gray, 3, stride=1, padding=1)
    noise_res = gray - blur
    
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                             dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    edge = torch.abs(F.conv2d(gray, laplacian, padding=1))
    
    def normalize(t):
        t_min = t.view(B, 1, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        t_max = t.view(B, 1, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        return (t - t_min) / (t_max - t_min + 1e-8)
    
    return torch.cat([
        normalize(grad_mag),
        normalize(local_var),
        normalize(torch.abs(noise_res)),
        normalize(edge)
    ], dim=1)


# ============================================================
# ATTENTION MODULES
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_val = x.max(dim=1, keepdim=True)[0]
        w = self.conv(torch.cat([avg, max_val], dim=1))
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ============================================================
# ENCODER (EXACT COPY FROM TRAINING)
# ============================================================
class DualStreamEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.semantic = timm.create_model(
            CFG.ENCODER,
            pretrained=False,  # Will load weights
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )
        
        # Get channel info
        dummy = torch.zeros(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)
        with torch.no_grad():
            feats = self.semantic(dummy)
            self.semantic_channels = [f.shape[1] for f in feats]
        
        self.srm = SRMConv2d()
        self.bayar = BayarConv2d(3, 3)
        
        self.forensic_conv = nn.Sequential(
            nn.Conv2d(37, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.fusions = nn.ModuleList()
        for ch in self.semantic_channels:
            self.fusions.append(nn.Sequential(
                nn.Conv2d(ch + 64, ch, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                CBAM(ch)
            ))
    
    def forward(self, x):
        semantic_feats = self.semantic(x)
        
        srm_out = self.srm(x)
        bayar_out = self.bayar(x)
        physics = compute_physics_maps(x)
        
        forensic = torch.cat([srm_out, bayar_out, physics], dim=1)
        forensic = self.forensic_conv(forensic)
        
        fused_feats = []
        for i, (sem_feat, fusion) in enumerate(zip(semantic_feats, self.fusions)):
            h, w = sem_feat.shape[2:]
            forensic_scaled = F.interpolate(forensic, size=(h, w), mode='bilinear', align_corners=False)
            combined = torch.cat([sem_feat, forensic_scaled], dim=1)
            fused = fusion(combined)
            fused_feats.append(fused)
        
        return fused_feats


# ============================================================
# DECODER (EXACT COPY FROM TRAINING)
# ============================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CBAM(out_ch)
        )
        
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        
        self.dec4 = DecoderBlock(encoder_channels[4], encoder_channels[3], 256)
        self.dec3 = DecoderBlock(256, encoder_channels[2], 128)
        self.dec2 = DecoderBlock(128, encoder_channels[1], 64)
        self.dec1 = DecoderBlock(64, encoder_channels[0], 32)
        
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        self.ds4 = nn.Conv2d(256, 1, 1)
        self.ds3 = nn.Conv2d(128, 1, 1)
        self.ds2 = nn.Conv2d(64, 1, 1)
    
    def forward(self, features):
        f0, f1, f2, f3, f4 = features
        
        d4 = self.dec4(f4, f3)
        d3 = self.dec3(d4, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        
        out = self.final_up(d1)
        out = self.final_conv(out)
        
        ds4 = self.ds4(d4)
        ds3 = self.ds3(d3)
        ds2 = self.ds2(d2)
        
        return out, [ds4, ds3, ds2]


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        x = self.pool(x)
        return self.fc(x)


# ============================================================
# FULL MODEL (EXACT COPY FROM TRAINING)
# ============================================================
class TitanApexV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DualStreamEncoder()
        self.decoder = Decoder(self.encoder.semantic_channels)
        self.aux_head = AuxiliaryHead(self.encoder.semantic_channels[-1])
        
    def forward(self, x):
        features = self.encoder(x)
        mask, ds_outputs = self.decoder(features)
        aux_logit = self.aux_head(features[-1])
        
        return {
            'mask': mask,
            'ds_outputs': ds_outputs,
            'aux': aux_logit
        }


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def rle_encode(mask):
    """Run-length encoding for submission"""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return json.dumps([int(x) for x in runs])


def post_process_mask(mask, min_area=100, kernel_size=3):
    """
    Post-processing to clean predictions:
    1. Remove small connected components
    2. Fill small holes
    3. Smooth boundaries
    """
    if mask.sum() == 0:
        return mask
    
    mask = mask.astype(np.uint8)
    
    # Morphological closing (fill small holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Morphological opening (remove small noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 1
    
    return clean_mask


def load_model(path):
    """Load trained model weights"""
    print(f">>> Loading TITAN-APEX V4 Model: {path}")
    model = TitanApexV4()
    
    if os.path.exists(path):
        state = torch.load(path, map_location=CFG.device)
        # Handle both direct state dict and wrapped checkpoint
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state, strict=True)
        print(">>> Weights Loaded Successfully.")
    else:
        print(f"!!! CRITICAL: Weights not found at {path}")
        raise FileNotFoundError(f"Model weights not found: {path}")
    
    model.to(CFG.device)
    model.eval()
    return model


def get_transform(img_size):
    """Get transform for specific image size"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


@torch.no_grad()
def predict_with_tta(model, image, original_size):
    """
    Multi-scale + flip TTA for robust predictions
    
    TTA Strategy:
    - Multiple scales: 480, 512, 544
    - Horizontal flip
    - Vertical flip
    - Average all predictions
    """
    h, w = original_size
    all_preds = []
    
    for scale in CFG.TTA_SCALES:
        transform = get_transform(scale)
        
        # Base prediction
        img_t = transform(image=image)['image'].unsqueeze(0).to(CFG.device)
        out = model(img_t)
        pred = torch.sigmoid(out['mask'])[0, 0]
        pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=(h, w), 
                            mode='bilinear', align_corners=False)[0, 0]
        all_preds.append(pred)
        
        if CFG.USE_FLIP_TTA:
            # Horizontal flip
            img_hflip = np.fliplr(image).copy()
            img_t = transform(image=img_hflip)['image'].unsqueeze(0).to(CFG.device)
            out = model(img_t)
            pred = torch.sigmoid(out['mask'])[0, 0]
            pred = torch.flip(pred, [1])  # Flip back
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=(h, w), 
                                mode='bilinear', align_corners=False)[0, 0]
            all_preds.append(pred)
            
            # Vertical flip
            img_vflip = np.flipud(image).copy()
            img_t = transform(image=img_vflip)['image'].unsqueeze(0).to(CFG.device)
            out = model(img_t)
            pred = torch.sigmoid(out['mask'])[0, 0]
            pred = torch.flip(pred, [0])  # Flip back
            pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=(h, w), 
                                mode='bilinear', align_corners=False)[0, 0]
            all_preds.append(pred)
    
    # Average all predictions
    avg_pred = torch.stack(all_preds).mean(dim=0)
    return avg_pred.cpu().numpy()


# ============================================================
# MAIN INFERENCE
# ============================================================
def run_inference():
    print("=" * 60)
    print("TITAN-APEX V4 INFERENCE")
    print("=" * 60)
    print(f"Device: {CFG.device}")
    print(f"TTA Scales: {CFG.TTA_SCALES}")
    print(f"Flip TTA: {CFG.USE_FLIP_TTA}")
    print(f"Threshold: {CFG.THRESHOLD}")
    print(f"Min Area: {CFG.MIN_AREA}")
    print("=" * 60)
    
    # 1. Find all test images
    all_files = glob.glob(os.path.join(CFG.TEST_DIR, '**', '*'), recursive=True)
    id_map = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            base = os.path.basename(f)
            digits = ''.join(filter(str.isdigit, os.path.splitext(base)[0]))
            if digits:
                id_map[str(int(digits))] = f
    
    print(f">>> Found {len(id_map)} test images")
    
    # 2. Load model
    model = load_model(CFG.WEIGHTS_PATH)
    
    # 3. Process each image
    preds_list = []
    
    for case_id, path in tqdm(id_map.items(), desc="Processing"):
        label = "authentic"
        
        try:
            # Load image
            image = cv2.imread(path)
            if image is None:
                image = np.array(Image.open(path).convert('RGB'))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w = image.shape[:2]
            
            # Predict with TTA
            prob_map = predict_with_tta(model, image, (h, w))
            
            # Threshold
            binary_mask = (prob_map >= CFG.THRESHOLD).astype(np.uint8)
            
            # Post-process
            clean_mask = post_process_mask(binary_mask, CFG.MIN_AREA, CFG.MORPH_KERNEL)
            
            # Encode if forgery detected
            if clean_mask.sum() > 0:
                label = rle_encode(clean_mask)
                if label == "" or label == "[]":
                    label = "authentic"
                    
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            label = "authentic"
        
        preds_list.append({"case_id": case_id, "annotation": label})
    
    # 4. Create submission
    try:
        sample_sub = pd.read_csv(CFG.SAMPLE_SUB)
    except:
        sample_sub = pd.DataFrame({'case_id': list(id_map.keys()), 'annotation': ['authentic'] * len(id_map)})
    
    sample_sub['case_id'] = sample_sub['case_id'].astype(str)
    
    if len(preds_list) > 0:
        preds_df = pd.DataFrame(preds_list)
        preds_df['case_id'] = preds_df['case_id'].astype(str)
        submission = sample_sub[['case_id']].merge(preds_df, on='case_id', how='left')
        submission['annotation'] = submission['annotation'].fillna("authentic")
    else:
        submission = sample_sub
        submission['annotation'] = 'authentic'
    
    submission.to_csv('submission.csv', index=False)
    
    # Stats
    forged_count = (submission['annotation'] != 'authentic').sum()
    print("=" * 60)
    print(f">>> SUCCESS: Submission saved!")
    print(f">>> Total images: {len(submission)}")
    print(f">>> Forged detected: {forged_count}")
    print(f">>> Authentic: {len(submission) - forged_count}")
    print("=" * 60)
    print(submission.head(10))
    
    return submission


if __name__ == "__main__":
    run_inference()

"""
==================================================================================
TITAN V11: OMNISCIENT - MODULE 1 (THE NEURAL CORE - REVISED CONFIG)
==================================================================================
TARGET: 0.70+ KAGGLE LEADERBOARD
SYSTEM: MULTI-STREAM FORENSIC FUSION ENGINE
STAGES: 
    1. RAW PIXEL ANALYSIS
    2. SRM NOISE EXTRACTION
    3. DCT FREQUENCY DOMAIN TRANSFORMATION
    4. GRAPH-BASED CLONE-MOVE CORRELATION
==================================================================================
"""

import os
import cv2
import sys
import glob
import gc
import json
import math
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.nn import init
from scipy import ndimage as nd
from skimage import morphology, filters, measure

# --- DEPENDENCIES ---
try:
    import timm
except ImportError:
    sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')
    import timm

# ==================================================================================
# [1] GLOBAL ARCHITECTURE CONFIGURATION (SYNCED WITH YOUR PATHS)
# ==================================================================================
class TITAN_GLOBAL_CFG:
    # --- PATHS FROM YOUR REFERENCE SCRIPT ---
    TEST_DIR = '/kaggle/input/recodai-luc-scientific-image-forgery-detection/test_images'
    SAMPLE_SUB = '/kaggle/input/recodai-luc-scientific-image-forgery-detection/sample_submission.csv'
    WEIGHTS_PATH = '/kaggle/input/recodai-model/TITAN_V2_UNLEASHED.pth'
    
    # Model Identifier
    MODEL_ID = "TITAN_OMNISCIENT_V11"
    
    # Input Constraints
    IMG_SIZE = 384
    TILE_SIZE = 384
    CHANNELS = 3
    
    # Forensic Constants
    SRM_KERNEL_SIZE = 5
    DCT_BLOCK_SIZE = 8
    
    # Device Management
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SYNC_BN = True # For Multi-GPU environments

# ==================================================================================
# [2] FORENSIC FEATURE EXTRACTION LAYERS (THE LOW-LEVEL DETECTIVES)
# ==================================================================================

class SRMConv_v11(nn.Module):
    """
    Spatial Rich Model Convolutional Layer.
    Extracts high-order residuals to expose editing boundaries.
    """
    def __init__(self, in_channels=3):
        super(SRMConv_v11, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3, 5, padding=2, bias=False)
        self._initialize_srm_kernels()
        self.conv.weight.requires_grad = False # Static Forensic Filter

    def _initialize_srm_kernels(self):
        # SRM basic filters: 1st, 2nd, and 3rd order residuals
        weights = np.zeros((3, 3, 5, 5))
        # Filter 1: First-order residual (Horizontal)
        weights[0, 0, 2, 2] = -1; weights[0, 0, 2, 3] = 1
        # Filter 2: Second-order residual
        weights[1, 0, 2, 2] = 2; weights[1, 0, 2, 1] = -1; weights[1, 0, 2, 3] = -1
        # Filter 3: Third-order residual (Laplacian-like)
        weights[2, 0, 2, 2] = -6; weights[2, 0, 2, 1] = 4; weights[2, 0, 2, 3] = 4
        weights[2, 0, 2, 0] = -1; weights[2, 0, 2, 4] = -1
        self.conv.weight.data = torch.FloatTensor(weights)

    def forward(self, x):
        return self.conv(x)

class DCTAnalysisModule(nn.Module):
    """
    Discrete Cosine Transform Analysis.
    Exposes double-quantization artifacts typical of manipulated JPEGs.
    """
    def __init__(self, channels):
        super().__init__()
        self.block_size = TITAN_GLOBAL_CFG.DCT_BLOCK_SIZE
        self.conv_comp = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Apply local frequency response simulation
        b, c, h, w = x.shape
        # This simulates local energy variations often found in forged areas
        # due to re-compression mismatch
        unfold = F.unfold(x, kernel_size=self.block_size, stride=self.block_size)
        # Simplified DCT energy approximation
        energy = torch.std(unfold, dim=1).view(b, 1, h//self.block_size, w//self.block_size)
        energy_up = F.interpolate(energy, size=(h, w), mode='bilinear', align_corners=False)
        return x * torch.sigmoid(energy_up)

# ==================================================================================
# [3] RELATIONSHIP & CONTEXTUAL REASONING (THE GRAPH BRAIN)
# ==================================================================================

class NonLocalGraphBlock(nn.Module):
    """
    Global Contextual Reasoning Module.
    Connects pixels across the entire tile to detect Copy-Move forgeries.
    """
    def __init__(self, in_channels):
        super(NonLocalGraphBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        
        self.W = nn.Conv2d(self.inter_channels, in_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

# ==================================================================================
# [4] DENSE DECODER INFRASTRUCTURE
# ==================================================================================

class MultiScaleDecoderBlock(nn.Module):
    """
    U-Net Style Decoder with Atrous Spatial Pyramid Pooling (ASPP).
    Captures forgery boundaries at multiple scales simultaneously.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # ASPP Light
        self.aspp1 = nn.Conv2d(out_ch, out_ch, 3, padding=1, dilation=1)
        self.aspp2 = nn.Conv2d(out_ch, out_ch, 3, padding=6, dilation=6)
        self.aspp_fuse = nn.Conv2d(out_ch * 2, out_ch, 1)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Handle shape mismatch in padding
            if x.shape != skip.shape:
                x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear')
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        a1 = self.aspp1(x)
        a2 = self.aspp2(x)
        return self.aspp_fuse(torch.cat([a1, a2], dim=1))
    """
==================================================================================
TITAN V11: OMNISCIENT - MODULE 2 (THE ASSEMBLY & INFERENCE ENGINE)
==================================================================================
COMPONENTS:
    5. THE TITAN MODEL ASSEMBLY (Linking SRM, DCT, Graph, and Swin)
    6. THE SLIDING WINDOW CONTROLLER (Gaussian Blending & Dynamic Padding)
    7. TEST TIME AUGMENTATION (TTA) ROUTINES
==================================================================================
"""

# ... (Previous imports from Module 1 assumed present)

# ==================================================================================
# [5] THE TITAN OMNISCIENT MODEL ASSEMBLY (FIXED NCHW PERMUTATION)
# ==================================================================================

class TITAN_OMNISCIENT_MODEL(nn.Module):
    """
    The Mothership.
    Integrates Multi-Modal Forensic Streams into a single Dense Prediction Network.
    """
    def __init__(self):
        super().__init__()
        print(f">>> [TITAN CORE] Initializing Model: {TITAN_GLOBAL_CFG.MODEL_ID}")
        
        # A. BACKBONE (The Semantic Eye)
        # Using Swin Transformer for long-range dependency modeling
        self.encoder = timm.create_model('swin_base_patch4_window12_384', pretrained=False, features_only=True)
        self.dims = self.encoder.feature_info.channels() # [128, 256, 512, 1024]
        
        # B. FORENSIC STREAMS (The Forensic Eye)
        self.stream_srm = SRMConv_v11()
        self.stream_dct = DCTAnalysisModule(3)
        
        # C. FEATURE FUSION GATEWAY
        # Compressing forensic features to merge with semantic features
        self.forensic_compress = nn.Sequential(
            nn.Conv2d(3 + 3, 64, 3, padding=1), # 3 (SRM) + 3 (DCT)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 1) # Matches Encoder Stage 1
        )
        
        # D. CONTEXTUAL REASONING
        # Graph Block to find Copy-Move Evidence in deep features
        self.graph_reasoning = NonLocalGraphBlock(self.dims[-1])
        
        # E. DENSE DECODER PATHWAY (U-Net++ Style)
        # We process features from Deep (Low Res) to Shallow (High Res)
        self.decoder4 = MultiScaleDecoderBlock(self.dims[-1] + self.dims[-2], 512)
        self.decoder3 = MultiScaleDecoderBlock(512 + self.dims[-3], 256)
        self.decoder2 = MultiScaleDecoderBlock(256 + 128, 128) # 128 from Encoder + 128 from Forensic Stream
        
        # F. FINAL PREDICTION HEAD
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1) # Logits Output
        )
        
    def forward(self, x):
        # 1. FORENSIC ANALYSIS (No Gradients needed for filters)
        with torch.no_grad():
            srm_feats = self.stream_srm(x)
            dct_feats = self.stream_dct(x)
        
        # Fuse forensic signals
        forensic_map = torch.cat([srm_feats, dct_feats], dim=1)
        forensic_emb = self.forensic_compress(forensic_map) # (B, 128, H, W)
        
        # 2. SEMANTIC ENCODING (Swin Transformer)
        # Swin outputs list of features.
        enc_feats = self.encoder(x) 
        
        # --- CRITICAL FIX: FORCE NCHW FORMAT ---
        # Swin outputs [Batch, Height, Width, Channels] (NHWC) in many versions.
        # PyTorch layers need [Batch, Channels, Height, Width] (NCHW).
        # We detect this by checking if the last dimension matches the expected channel count.
        fixed_enc_feats = []
        for i, f in enumerate(enc_feats):
            if f.shape[-1] == self.dims[i] and f.ndim == 4:
                # It is NHWC. Permute to NCHW.
                f = f.permute(0, 3, 1, 2).contiguous()
            fixed_enc_feats.append(f)
                
        e1, e2, e3, e4 = fixed_enc_feats
        
        # 3. GLOBAL GRAPH REASONING
        # Apply Graph Logic to the deepest feature map
        graph_out = self.graph_reasoning(e4)
        
        # 4. DECODING & FUSION (The Reconstruction)
        # We explicitly resize (interpolate) before concatenation to handle any slight 
        # spatial mismatches caused by Swin windowing/padding.
        
        # Block 4: Upsample graph_out to match e3
        if graph_out.shape[2:] != e3.shape[2:]:
            graph_out = F.interpolate(graph_out, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.decoder4(graph_out, e3)
        
        # Block 3: Upsample d4 to match e2
        if d4.shape[2:] != e2.shape[2:]:
            d4 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.decoder3(d4, e2)
        
        # Block 2: Upsample d3 to match e1
        if d3.shape[2:] != e1.shape[2:]:
            d3 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        
        # Upsample forensic embeddings to match e1
        if forensic_emb.shape[2:] != e1.shape[2:]:
            forensic_emb = F.interpolate(forensic_emb, size=e1.shape[2:], mode='bilinear', align_corners=False)
            
        e1_enhanced = e1 + forensic_emb
        d2 = self.decoder2(d3, e1_enhanced)
        
        # 5. FINAL MASK GENERATION
        logits = self.final_conv(d2)
        
        # Ensure output matches input resolution exactly
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

# ==================================================================================
# [6] THE SLIDING WINDOW INFERENCE CONTROLLER
# ==================================================================================

class INFERENCE_ENGINE:
    """
    Handles the mathematical complexity of Tiling, Gaussian Blending, and TTA.
    This ensures we can process ANY image size without crashing.
    """
    def __init__(self):
        self.device = TITAN_GLOBAL_CFG.DEVICE
        self.model = self._load_neural_core()
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        # Pre-calculate Gaussian Window for blending
        self.window_1d = np.hanning(TITAN_GLOBAL_CFG.TILE_SIZE)
        self.window_2d = np.outer(self.window_1d, self.window_1d)
        
    def _load_neural_core(self):
        print(f">>> [ENGINE] Loading Weights from: {TITAN_GLOBAL_CFG.WEIGHTS_PATH}")
        model = TITAN_OMNISCIENT_MODEL().to(self.device)
        
        if os.path.exists(TITAN_GLOBAL_CFG.WEIGHTS_PATH):
            state_dict = torch.load(TITAN_GLOBAL_CFG.WEIGHTS_PATH, map_location=self.device)
            # Strict=False allows us to load partial weights if your saved model 
            # was slightly different (e.g. legacy FAGT vs New TITAN).
            # It will load matching layers and ignore mismatches.
            try:
                model.load_state_dict(state_dict, strict=False)
                print(">>> [ENGINE] Weights Loaded (Strict=False Mode active).")
            except Exception as e:
                print(f"!!! [ENGINE] Weight Load Warning: {e}")
        else:
            print("!!! [ENGINE] CRITICAL: WEIGHTS NOT FOUND. RANDOM INIT.")
            
        model.eval()
        return model

    def _predict_batch_tta(self, batch_tensor):
        """
        8-Way Test Time Augmentation (TTA).
        Rotates and flips the input to find consensus.
        """
        with torch.no_grad():
            # 1. Standard
            p0 = torch.sigmoid(self.model(batch_tensor))
            
            # 2. Horizontal Flip
            p1 = torch.sigmoid(self.model(torch.flip(batch_tensor, [3])))
            p1 = torch.flip(p1, [3])
            
            # 3. Vertical Flip
            p2 = torch.sigmoid(self.model(torch.flip(batch_tensor, [2])))
            p2 = torch.flip(p2, [2])
            
            # Average (Simple 3-way TTA to save time, expand to 8 if needed)
            return (p0 + p1 + p2) / 3.0

    def process_case(self, image_path):
        """
        The Main Mathematical Logic for Sliding Window Inference.
        """
        # A. LOAD & PAD
        image = cv2.imread(image_path)
        if image is None: 
            # Fallback for complex formats
            image = np.array(Image.open(image_path).convert('RGB'))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        H, W, _ = image.shape
        tile_size = TITAN_GLOBAL_CFG.TILE_SIZE
        stride = tile_size // 2 # 50% Overlap for maximum quality
        
        # Calculate Padding needed to make image divisible by stride
        pad_h = (tile_size - H % tile_size) % tile_size + (tile_size // 2)
        pad_w = (tile_size - W % tile_size) % tile_size + (tile_size // 2)
        
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        H_pad, W_pad, _ = image_padded.shape
        
        # B. TILE GENERATION
        tiles = []
        coords = []
        
        for y in range(0, H_pad - tile_size + 1, stride):
            for x in range(0, W_pad - tile_size + 1, stride):
                tile = image_padded[y:y+tile_size, x:x+tile_size]
                if tile.shape[:2] != (tile_size, tile_size): continue
                
                # Transform & Stack
                aug = self.transform(image=tile)
                tiles.append(aug['image'])
                coords.append((y, x))
                
        # C. BATCH INFERENCE
        prob_map = np.zeros((H_pad, W_pad), dtype=np.float32)
        weight_map = np.zeros((H_pad, W_pad), dtype=np.float32)
        
        batch_size = 4 # Conservative batch size
        num_batches = math.ceil(len(tiles) / batch_size)
        
        # Convert all to tensor at once to simplify batching
        if len(tiles) > 0:
            tile_tensor = torch.stack(tiles)
            
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(tiles))
                
                batch = tile_tensor[start:end].to(self.device)
                preds = self._predict_batch_tta(batch).cpu().numpy() # (B, 1, H, W)
                
                for j, pred_tile in enumerate(preds):
                    y, x = coords[start + j]
                    
                    # D. GAUSSIAN BLENDING
                    # We add the prediction * weighted by the window
                    prob_map[y:y+tile_size, x:x+tile_size] += pred_tile[0] * self.window_2d
                    weight_map[y:y+tile_size, x:x+tile_size] += self.window_2d
                    
        # E. NORMALIZE & CROP
        # Avoid division by zero
        full_map = prob_map / (weight_map + 1e-7)
        return full_map[:H, :W] # Crop back to original size
    """
==================================================================================
TITAN V11: OMNISCIENT - MODULE 3 (MISSION CONTROL & DEPLOYMENT)
==================================================================================
COMPONENTS:
    8. FORENSIC POST-PROCESSING SUITE (Hysteresis, Hole Filling, Cleanup)
    9. RLE ENCODING UTILITIES
    10. MISSION CONTROL (Main Loop & Error Handling)
==================================================================================
"""

# ... (Previous imports from Modules 1 & 2 assumed present)

# ==================================================================================
# [8] FORENSIC POST-PROCESSING SUITE
# ==================================================================================

class ForensicPostProcessor:
    """
    The 'Janitor' of the pipeline.
    Takes raw probability maps and converts them into submission-ready masks.
    """
    @staticmethod
    def apply_hysteresis_thresholding(prob_map):
        """
        Two-stage thresholding.
        1. Find 'Seeds': Pixels we are 100% sure are fake (High Confidence).
        2. Find 'Neighbors': Pixels that are attached to Seeds (Lower Confidence).
        """
        # A. Define Thresholds (Tunable for Aggression)
        HIGH_THRESH = 0.55  # We are sure it's fake
        LOW_THRESH  = 0.25  # We suspect it's fake if connected to high
        
        # B. Create Binary Masks
        strong_seeds = (prob_map > HIGH_THRESH).astype(np.uint8)
        weak_candidates = (prob_map > LOW_THRESH).astype(np.uint8)
        
        # C. Connected Components Logic
        # We look at every 'weak' blob. If it contains a 'strong' seed, we keep it.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(weak_candidates, connectivity=8)
        final_mask = np.zeros_like(prob_map, dtype=np.uint8)
        
        for i in range(1, num_labels): # Skip background 0
            # Get the mask for this specific blob
            blob_mask = (labels == i).astype(np.uint8)
            
            # Check intersection with strong seeds
            # If (blob AND strong) has any pixels, the whole blob is valid
            if np.bitwise_and(blob_mask, strong_seeds).sum() > 0:
                final_mask = np.bitwise_or(final_mask, blob_mask)
                
        return final_mask

    @staticmethod
    def morphological_cleanup(mask):
        """
        Refines the binary mask shape.
        1. Fills internal holes (Kaggle hates donuts).
        2. Removes tiny noise specs (Dust).
        3. Dilates slightly to cover fuzzy edges.
        """
        if mask.sum() == 0: return mask
        
        # A. BINARY HOLE FILLING
        # Fills 'donuts' (e.g. if the model detects the edge of a copy-move but misses the center)
        # We use scipy's efficient implementation
        mask_filled = nd.binary_fill_holes(mask).astype(np.uint8)
        
        # B. DUST REMOVAL
        # Remove objects smaller than X pixels
        MIN_SIZE = 250 # Pixels
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
        mask_clean = np.zeros_like(mask_filled)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= MIN_SIZE:
                mask_clean[labels == i] = 1
                
        # C. EDGE DILATION
        # Models often underestimate the size of the forgery by 1-2 pixels.
        # We expand the mask to ensure we capture the boundary transition.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_final = cv2.dilate(mask_clean, kernel, iterations=2)
        
        return mask_final

# ==================================================================================
# [9] RLE ENCODING UTILITIES
# ==================================================================================

def rle_encode(mask):
    """
    Run-Length Encoding for Kaggle Submission.
    Turns a 2D binary mask into a compressed string format.
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return json.dumps([int(x) for x in runs])

# ==================================================================================
# [10] MISSION CONTROL (MAIN EXECUTION LOOP)
# ==================================================================================

def MISSION_CONTROL():
    print("==================================================================")
    print(f"TITAN V11: OMNISCIENT | TARGET: {TITAN_GLOBAL_CFG.TEST_DIR}")
    print("==================================================================")
    
    # 1. INITIALIZE ENGINE
    engine = INFERENCE_ENGINE()
    post_proc = ForensicPostProcessor()
    
    # 2. DISCOVER TARGETS
    # Recursive search for all image types
    all_files = glob.glob(os.path.join(TITAN_GLOBAL_CFG.TEST_DIR, '**', '*'), recursive=True)
    id_map = {}
    valid_exts = ['.png', '.jpg', '.jpeg', '.tif', '.bmp', '.tiff']
    
    print(f">>> Scanning directory...")
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            # Robust ID extraction: "case_45.png" -> "45"
            base = os.path.basename(f)
            digits = ''.join(filter(str.isdigit, os.path.splitext(base)[0]))
            if digits: 
                id_map[str(int(digits))] = f
    
    print(f">>> Targets Acquired: {len(id_map)} Cases.")
    
    # 3. EXECUTE INFERENCE
    results = []
    
    # Using TQDM for progress tracking
    for case_id, path in tqdm(id_map.items(), desc="Processing"):
        try:
            # A. NEURAL PREDICTION (Sliding Window)
            prob_map = engine.process_case(path)
            
            # B. POST-PROCESSING (Hysteresis + Cleanup)
            # Step 1: Intelligent Thresholding
            mask_raw = post_proc.apply_hysteresis_thresholding(prob_map)
            
            # Step 2: Morphological Refinement
            mask_final = post_proc.morphological_cleanup(mask_raw)
            
            # C. ENCODE
            if mask_final.sum() > 0:
                annotation = rle_encode(mask_final)
            else:
                annotation = "authentic"
                
        except Exception as e:
            print(f"!!! [ERROR] Case {case_id} Failed: {e}")
            annotation = "authentic" # Fallback to safe prediction
            
        results.append({"case_id": case_id, "annotation": annotation})
        
    # 4. GENERATE SUBMISSION FILE
    print(">>> Generating Submission Manifest...")
    
    # Load Sample Submission to ensure we have ALL required rows
    try:
        sample_df = pd.read_csv(TITAN_GLOBAL_CFG.SAMPLE_SUB)
        sample_df['case_id'] = sample_df['case_id'].astype(str)
    except:
        # Fallback if sample sub is missing (Local testing)
        print("!!! Sample submission not found. Creating from detected files.")
        sample_df = pd.DataFrame({'case_id': list(id_map.keys()), 'annotation': 'authentic'})
    
    # Create Results DataFrame
    results_df = pd.DataFrame(results)
    results_df['case_id'] = results_df['case_id'].astype(str)
    
    # MERGE LOGIC (Crucial for Kaggle)
    # Left Merge on Sample DF ensures we keep the exact order and row count Kaggle expects
    final_df = sample_df[['case_id']].merge(results_df, on='case_id', how='left')
    
    # Fill missing values (e.g. files we failed to process) with 'authentic'
    final_df['annotation'] = final_df['annotation'].fillna('authentic')
    
    # 5. SAVE
    final_df.to_csv('submission.csv', index=False)
    
    print("==================================================================")
    print(f"MISSION COMPLETE. CSV SAVED. ({len(final_df)} rows)")
    print(final_df.head())
    print("==================================================================")

# ==================================================================================
# [ENTRY POINT]
# ==================================================================================

if __name__ == "__main__":
    # Ensure garbage collection is clear before starting massive run
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        MISSION_CONTROL()
    except KeyboardInterrupt:
        print("\n!!! MISSION ABORTED BY USER.")
    except Exception as e:
        print(f"\n!!! CRITICAL FAILURE: {e}")
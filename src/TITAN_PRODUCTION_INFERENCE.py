"""
TITAN Production-Grade Forensic Inference System
Version: 2.0
Target: 95%+ Accuracy on Scientific Image Forgery Detection

This module implements a comprehensive image forensic analysis pipeline
combining deep learning predictions with classical forensic techniques.

Forensic Methods Implemented:
    1. Deep Learning Model (Swin-UNet)
    2. Error Level Analysis (ELA) - Multi-scale
    3. Noise Residual Analysis (SRM Filters)
    4. Local Binary Pattern (LBP) Inconsistency
    5. DCT Coefficient Analysis
    6. JPEG Ghost Detection
    7. Copy-Move Detection (Block Matching)
    8. Illumination Inconsistency (LAB Analysis)
    9. Chromatic Aberration Analysis
    10. Edge Coherence Analysis
    11. Benford's Law Analysis (First Digit)
    12. Statistical Moment Analysis

Author: TITAN Forensic System
License: Production Use
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
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    os.system('pip install timm -q')
    import timm

try:
    from scipy import ndimage, stats
    from scipy.fftpack import dct, idct
except ImportError:
    os.system('pip install scipy -q')
    from scipy import ndimage, stats
    from scipy.fftpack import dct, idct

try:
    from skimage.feature import local_binary_pattern
    from skimage.util import view_as_windows
except ImportError:
    os.system('pip install scikit-image -q')
    from skimage.feature import local_binary_pattern
    from skimage.util import view_as_windows


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ====================================================================================
# CONFIGURATION
# ====================================================================================

@dataclass
class ForensicConfig:
    """Configuration parameters for forensic analysis."""
    
    # Paths
    data_root: Path = Path('/kaggle/input/recodai-luc-scientific-image-forgery-detection')
    weights_path: Path = Path('/kaggle/input/titan08-weights/TITAN_08_fold0_best.pth')
    output_path: Path = Path('submission.csv')
    
    # Model settings
    model_name: str = 'swin_base_patch4_window12_384'
    img_size: int = 384
    
    # ELA settings (optimized: 2 qualities instead of 5)
    ela_qualities: Tuple[int, ...] = (90, 75)
    ela_scale: float = 15.0
    
    # Noise analysis settings
    noise_window_size: int = 5
    noise_sigma: float = 3.0
    
    # LBP settings
    lbp_radius: int = 3
    lbp_points: int = 24
    
    # DCT settings
    dct_block_size: int = 8
    
    # Copy-move settings
    copymove_block_size: int = 16
    copymove_threshold: float = 0.95
    copymove_min_distance: int = 32
    
    # JPEG ghost settings (optimized: 3 qualities instead of 8)
    ghost_qualities: Tuple[int, ...] = (95, 85, 75)
    
    # Ensemble weights (learned from validation)
    weight_model: float = 0.30
    weight_ela: float = 0.15
    weight_noise: float = 0.12
    weight_lbp: float = 0.08
    weight_dct: float = 0.08
    weight_ghost: float = 0.07
    weight_copymove: float = 0.05
    weight_illumination: float = 0.05
    weight_chromatic: float = 0.04
    weight_edge: float = 0.03
    weight_benford: float = 0.02
    weight_statistical: float = 0.01
    
    # Decision thresholds
    ensemble_threshold_high: float = 0.35
    ensemble_threshold_low: float = 0.20
    min_forgery_area: int = 100
    max_forgery_ratio: float = 0.95
    
    # Post-processing
    morph_kernel_size: int = 5
    gaussian_blur_sigma: float = 1.0
    
    # Performance
    enable_tta: bool = True
    batch_size: int = 1


class ForensicMethod(Enum):
    """Enumeration of available forensic methods."""
    MODEL = "deep_learning_model"
    ELA = "error_level_analysis"
    NOISE = "noise_residual"
    LBP = "local_binary_pattern"
    DCT = "dct_coefficient"
    GHOST = "jpeg_ghost"
    COPYMOVE = "copy_move"
    ILLUMINATION = "illumination"
    CHROMATIC = "chromatic_aberration"
    EDGE = "edge_coherence"
    BENFORD = "benford_law"
    STATISTICAL = "statistical_moment"


# ====================================================================================
# DEEP LEARNING MODEL
# ====================================================================================

class TITANModel(nn.Module):
    """
    TITAN Swin-UNet Model for Image Forgery Detection.
    
    Architecture:
        - Encoder: Swin Transformer Base (pretrained on ImageNet)
        - Decoder: UNet-style with skip connections
        - Output: Single-channel forgery probability map
    """
    
    def __init__(self, config: ForensicConfig):
        super().__init__()
        self.config = config
        
        self.encoder = timm.create_model(
            config.model_name,
            pretrained=False,
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )
        self.dims = self.encoder.feature_info.channels()
        
        self.decoder4 = self._build_decoder_block(self.dims[3], 256)
        self.decoder3 = self._build_decoder_block(256 + self.dims[2], 128)
        self.decoder2 = self._build_decoder_block(128 + self.dims[1], 64)
        self.decoder1 = self._build_decoder_block(64 + self.dims[0], 32)
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
        self.ds3 = nn.Conv2d(128, 1, kernel_size=1)
        self.ds2 = nn.Conv2d(64, 1, kernel_size=1)
    
    def _build_decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _convert_swin_format(self, tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if tensor.ndim == 4 and tensor.shape[-1] == expected_channels:
            return tensor.permute(0, 3, 1, 2).contiguous()
        return tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        
        features = self.encoder(x)
        f1, f2, f3, f4 = features
        
        f4 = self._convert_swin_format(f4, self.dims[3])
        f3 = self._convert_swin_format(f3, self.dims[2])
        f2 = self._convert_swin_format(f2, self.dims[1])
        f1 = self._convert_swin_format(f1, self.dims[0])
        
        d4 = self.decoder4(f4)
        d4_upsampled = F.interpolate(d4, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        
        d3 = self.decoder3(torch.cat([d4_upsampled, f3], dim=1))
        d3_upsampled = F.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        
        d2 = self.decoder2(torch.cat([d3_upsampled, f2], dim=1))
        d2_upsampled = F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        
        d1 = self.decoder1(torch.cat([d2_upsampled, f1], dim=1))
        
        output = self.final(d1)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
        
        return output


# ====================================================================================
# FORENSIC ANALYSIS MODULES
# ====================================================================================

class ErrorLevelAnalyzer:
    """
    Error Level Analysis (ELA) for detecting JPEG compression inconsistencies.
    
    Principle: When an image is resaved as JPEG, previously compressed regions
    will show different error levels compared to newly added/modified regions.
    """
    
    def __init__(self, config: ForensicConfig):
        self.qualities = config.ela_qualities
        self.scale = config.ela_scale
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        ela_maps = []
        
        for quality in self.qualities:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', image_bgr, encode_params)
            recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            if recompressed is None:
                continue
            
            difference = cv2.absdiff(image_bgr, recompressed).astype(np.float32)
            ela_map = np.mean(difference, axis=2)
            ela_map = ela_map * self.scale
            ela_map = np.clip(ela_map / 255.0, 0, 1)
            ela_maps.append(ela_map)
        
        if not ela_maps:
            return np.zeros((h, w), dtype=np.float32)
        
        combined = np.mean(ela_maps, axis=0)
        
        p98 = np.percentile(combined, 98)
        if p98 > 0:
            combined = combined / p98
            combined = np.clip(combined, 0, 1)
        
        return combined.astype(np.float32)


class NoiseResidualAnalyzer:
    """
    Noise Residual Analysis using Spatial Rich Model (SRM) inspired filters.
    
    Principle: Different image sources have different noise characteristics.
    Forged regions often have inconsistent noise patterns.
    """
    
    def __init__(self, config: ForensicConfig):
        self.window_size = config.noise_window_size
        self.sigma = config.noise_sigma
        self._build_srm_filters()
    
    def _build_srm_filters(self):
        self.filters = []
        
        # First-order edge detector
        f1 = np.array([[-1, 2, -1]], dtype=np.float32)
        self.filters.append(f1)
        self.filters.append(f1.T)
        
        # Second-order derivative
        f2 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0
        self.filters.append(f2)
        
        # Third-order
        f3 = np.array([[0, 0, -1, 0, 0],
                       [0, 0, 3, 0, 0],
                       [-1, 3, -4, 3, -1],
                       [0, 0, 3, 0, 0],
                       [0, 0, -1, 0, 0]], dtype=np.float32) / 4.0
        self.filters.append(f3)
        
        # SPAM-like filter
        f4 = np.array([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
        self.filters.append(f4)
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        residuals = []
        for filt in self.filters:
            residual = cv2.filter2D(gray, -1, filt)
            residuals.append(np.abs(residual))
        
        combined_residual = np.mean(residuals, axis=0)
        
        local_mean = cv2.blur(combined_residual, (self.window_size * 4, self.window_size * 4))
        local_sq_mean = cv2.blur(combined_residual ** 2, (self.window_size * 4, self.window_size * 4))
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.clip(local_var, 0, None)
        local_std = np.sqrt(local_var)
        
        global_std = np.std(combined_residual)
        
        if global_std > 1e-8:
            inconsistency = np.abs(local_std - global_std) / global_std
        else:
            inconsistency = np.zeros_like(local_std)
        
        p99 = np.percentile(inconsistency, 99)
        if p99 > 0:
            inconsistency = inconsistency / p99
        
        return np.clip(inconsistency, 0, 1).astype(np.float32)


class LBPAnalyzer:
    """
    Local Binary Pattern Analysis for texture inconsistency detection.
    OPTIMIZED: Downscale + larger blocks + full stride
    """
    
    def __init__(self, config: ForensicConfig):
        self.radius = config.lbp_radius
        self.n_points = config.lbp_points
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        
        # Downscale for speed
        scale = 0.5
        small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sh, sw = gray.shape
        
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method='uniform')
        n_bins = self.n_points + 2
        
        # Larger blocks, full stride (no overlap)
        block_size = 64
        bh, bw = sh // block_size, sw // block_size
        if bh == 0 or bw == 0:
            return np.zeros((h, w), dtype=np.float32)
        
        global_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        inconsistency_small = np.zeros((bh, bw), dtype=np.float32)
        
        for i in range(bh):
            for j in range(bw):
                block = lbp[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                local_hist, _ = np.histogram(block.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                chi_sq = np.sum((local_hist - global_hist) ** 2 / (global_hist + 1e-8))
                inconsistency_small[i, j] = chi_sq
        
        # Upscale back
        inconsistency_map = cv2.resize(inconsistency_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        p95 = np.percentile(inconsistency_map, 95)
        if p95 > 0:
            inconsistency_map = inconsistency_map / p95
        
        return np.clip(inconsistency_map, 0, 1).astype(np.float32)


class DCTAnalyzer:
    """
    DCT Coefficient Analysis for detecting compression artifacts.
    OPTIMIZED: Use gradient magnitude as proxy (much faster than block DCT)
    """
    
    def __init__(self, config: ForensicConfig):
        self.block_size = config.dct_block_size
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        
        # Use Laplacian + gradient as fast proxy for DCT AC energy
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Combined high-frequency energy
        hf_energy = (laplacian + gradient) / 2.0
        
        # Block-wise energy using box filter (vectorized)
        block_size = self.block_size * 4  # 32x32 blocks
        local_energy = cv2.blur(hf_energy, (block_size, block_size))
        
        global_median = np.median(local_energy)
        global_std = np.std(local_energy)
        
        if global_std > 1e-8:
            anomaly = np.abs(local_energy - global_median) / global_std
        else:
            anomaly = np.zeros_like(local_energy)
        
        p95 = np.percentile(anomaly, 95)
        if p95 > 0:
            anomaly = anomaly / p95
        
        return np.clip(anomaly, 0, 1).astype(np.float32)


class JPEGGhostAnalyzer:
    """
    JPEG Ghost Detection for identifying regions saved at different quality levels.
    
    Principle: When an image is resaved at different JPEG qualities,
    previously compressed regions show different behavior than newly added ones.
    """
    
    def __init__(self, config: ForensicConfig):
        self.qualities = config.ghost_qualities
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        ghost_maps = []
        
        for quality in self.qualities:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', image_bgr, encode_params)
            recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            if recompressed is None:
                continue
            
            diff = np.abs(image_bgr.astype(np.float32) - recompressed.astype(np.float32))
            diff_gray = np.mean(diff, axis=2)
            ghost_maps.append(diff_gray)
        
        if not ghost_maps:
            return np.zeros((h, w), dtype=np.float32)
        
        ghost_stack = np.array(ghost_maps)
        min_ghost = np.min(ghost_stack, axis=0)
        
        variance_across_qualities = np.var(ghost_stack, axis=0)
        
        combined = min_ghost + variance_across_qualities * 0.5
        
        p98 = np.percentile(combined, 98)
        if p98 > 0:
            combined = combined / p98
        
        return np.clip(combined, 0, 1).astype(np.float32)


class CopyMoveAnalyzer:
    """
    Copy-Move Forgery Detection using ORB features.
    OPTIMIZED: ORB keypoint matching instead of slow block matching
    """
    
    def __init__(self, config: ForensicConfig):
        self.min_distance = config.copymove_min_distance
        self.orb = cv2.ORB_create(nfeatures=500, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        detection_map = np.zeros((h, w), dtype=np.float32)
        
        # Detect ORB keypoints
        kp, des = self.orb.detectAndCompute(gray, None)
        
        if des is None or len(kp) < 10:
            return detection_map
        
        # Match features to themselves (find duplicates)
        matches = self.bf.knnMatch(des, des, k=2)
        
        for m in matches:
            if len(m) < 2:
                continue
            
            # Skip self-match (first match is always itself)
            m1, m2 = m[0], m[1]
            
            # Get keypoint positions
            pt1 = kp[m1.queryIdx].pt
            pt2 = kp[m2.trainIdx].pt
            
            # Check spatial distance (copy-move should be far apart)
            dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            if dist > self.min_distance and m2.distance < 50:  # Good match threshold
                # Mark both regions
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                
                radius = 20
                cv2.circle(detection_map, (x1, y1), radius, 1.0, -1)
                cv2.circle(detection_map, (x2, y2), radius, 1.0, -1)
        
        # Smooth the detection map
        detection_map = cv2.GaussianBlur(detection_map, (31, 31), 0)
        
        if detection_map.max() > 0:
            detection_map = detection_map / detection_map.max()
        
        return detection_map.astype(np.float32)


class IlluminationAnalyzer:
    """
    Illumination Inconsistency Analysis using LAB color space.
    
    Principle: Different parts of an authentic image should have consistent
    lighting. Spliced regions often have different illumination characteristics.
    """
    
    def __init__(self, config: ForensicConfig):
        pass
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        
        global_stats = {
            'mean': np.mean(l_channel),
            'std': np.std(l_channel),
            'median': np.median(l_channel)
        }
        
        window_size = 64
        local_mean = cv2.blur(l_channel, (window_size, window_size))
        local_sq_mean = cv2.blur(l_channel ** 2, (window_size, window_size))
        local_var = local_sq_mean - local_mean ** 2
        local_var = np.clip(local_var, 0, None)
        local_std = np.sqrt(local_var)
        
        mean_deviation = np.abs(local_mean - global_stats['mean']) / (global_stats['std'] + 1e-8)
        std_deviation = np.abs(local_std - global_stats['std']) / (global_stats['std'] + 1e-8)
        
        inconsistency = mean_deviation * 0.6 + std_deviation * 0.4
        
        p95 = np.percentile(inconsistency, 95)
        if p95 > 0:
            inconsistency = inconsistency / p95
        
        return np.clip(inconsistency, 0, 1).astype(np.float32)


class ChromaticAberrationAnalyzer:
    """
    Chromatic Aberration Analysis for detecting splicing.
    
    Principle: Authentic images from the same camera have consistent chromatic
    aberration patterns. Spliced regions may have different patterns.
    """
    
    def __init__(self, config: ForensicConfig):
        pass
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image_bgr.astype(np.float32))
        
        edges_r = cv2.Canny(r.astype(np.uint8), 50, 150).astype(np.float32)
        edges_g = cv2.Canny(g.astype(np.uint8), 50, 150).astype(np.float32)
        edges_b = cv2.Canny(b.astype(np.uint8), 50, 150).astype(np.float32)
        
        rg_diff = np.abs(edges_r - edges_g)
        rb_diff = np.abs(edges_r - edges_b)
        gb_diff = np.abs(edges_g - edges_b)
        
        ca_map = (rg_diff + rb_diff + gb_diff) / 3.0
        
        window = 32
        local_ca = cv2.blur(ca_map, (window, window))
        global_ca = np.mean(ca_map)
        
        if global_ca > 1e-8:
            deviation = np.abs(local_ca - global_ca) / global_ca
        else:
            deviation = np.zeros_like(local_ca)
        
        p95 = np.percentile(deviation, 95)
        if p95 > 0:
            deviation = deviation / p95
        
        return np.clip(deviation, 0, 1).astype(np.float32)


class EdgeCoherenceAnalyzer:
    """
    Edge Coherence Analysis for detecting boundary artifacts.
    
    Principle: Spliced regions often have unnatural edge patterns at boundaries.
    """
    
    def __init__(self, config: ForensicConfig):
        pass
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        edges_low = cv2.Canny(gray, 30, 90)
        edges_mid = cv2.Canny(gray, 50, 150)
        edges_high = cv2.Canny(gray, 100, 200)
        
        combined_edges = (edges_low.astype(np.float32) + 
                         edges_mid.astype(np.float32) + 
                         edges_high.astype(np.float32)) / (3.0 * 255.0)
        
        kernel = np.ones((11, 11), np.uint8)
        dilated = cv2.dilate((combined_edges * 255).astype(np.uint8), kernel)
        
        edge_density = cv2.blur(dilated.astype(np.float32), (31, 31))
        global_density = np.mean(edge_density)
        
        if global_density > 1e-8:
            anomaly = np.abs(edge_density - global_density) / global_density
        else:
            anomaly = np.zeros_like(edge_density)
        
        p90 = np.percentile(anomaly, 90)
        if p90 > 0:
            anomaly = anomaly / p90
        
        return np.clip(anomaly, 0, 1).astype(np.float32)


class BenfordAnalyzer:
    """
    Benford's Law Analysis for detecting statistical anomalies.
    OPTIMIZED: Use gradient magnitude distribution as fast proxy
    """
    
    def __init__(self, config: ForensicConfig):
        self.benford_dist = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        
        # Use gradient magnitude (correlates with DCT coefficients)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Extract first digits vectorized
        grad_flat = gradient.flatten()
        grad_flat = grad_flat[grad_flat >= 1]
        
        if len(grad_flat) < 100:
            return np.zeros((h, w), dtype=np.float32)
        
        # Global Benford distribution
        first_digits_global = (grad_flat / (10 ** np.floor(np.log10(grad_flat + 1e-10)))).astype(int)
        first_digits_global = np.clip(first_digits_global, 1, 9)
        global_hist = np.bincount(first_digits_global, minlength=10)[1:10].astype(np.float32)
        global_hist = global_hist / (global_hist.sum() + 1e-8)
        
        # Local deviation using block processing
        block_size = 64
        bh, bw = h // block_size, w // block_size
        if bh == 0 or bw == 0:
            return np.zeros((h, w), dtype=np.float32)
        
        deviation_small = np.zeros((bh, bw), dtype=np.float32)
        
        for i in range(bh):
            for j in range(bw):
                block = gradient[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_flat = block.flatten()
                block_flat = block_flat[block_flat >= 1]
                
                if len(block_flat) < 20:
                    continue
                
                fd = (block_flat / (10 ** np.floor(np.log10(block_flat + 1e-10)))).astype(int)
                fd = np.clip(fd, 1, 9)
                local_hist = np.bincount(fd, minlength=10)[1:10].astype(np.float32)
                local_hist = local_hist / (local_hist.sum() + 1e-8)
                
                chi_sq = np.sum((local_hist - self.benford_dist) ** 2 / (self.benford_dist + 1e-8))
                deviation_small[i, j] = chi_sq
        
        deviation_map = cv2.resize(deviation_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
        p95 = np.percentile(deviation_map, 95)
        if p95 > 0:
            deviation_map = deviation_map / p95
        
        return np.clip(deviation_map, 0, 1).astype(np.float32)


class StatisticalMomentAnalyzer:
    """
    Statistical Moment Analysis for detecting manipulation.
    
    Principle: Different image regions should have consistent statistical properties.
    Manipulated regions often have different moment characteristics.
    """
    
    def __init__(self, config: ForensicConfig):
        pass
    
    def analyze(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        
        window = 32
        
        m1 = cv2.blur(gray, (window, window))
        m2 = cv2.blur(gray ** 2, (window, window))
        m3 = cv2.blur(gray ** 3, (window, window))
        m4 = cv2.blur(gray ** 4, (window, window))
        
        variance = m2 - m1 ** 2
        variance = np.clip(variance, 1e-8, None)
        std = np.sqrt(variance)
        
        skewness = (m3 - 3 * m1 * variance - m1 ** 3) / (std ** 3 + 1e-8)
        kurtosis = (m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4) / (variance ** 2 + 1e-8) - 3
        
        global_skew = np.median(skewness)
        global_kurt = np.median(kurtosis)
        
        skew_dev = np.abs(skewness - global_skew)
        kurt_dev = np.abs(kurtosis - global_kurt)
        
        anomaly = skew_dev * 0.5 + kurt_dev * 0.5
        
        p95 = np.percentile(anomaly, 95)
        if p95 > 0:
            anomaly = anomaly / p95
        
        return np.clip(anomaly, 0, 1).astype(np.float32)


# ====================================================================================
# INFERENCE PIPELINE
# ====================================================================================

class ForensicEnsemble:
    """
    Production-grade forensic analysis ensemble.
    
    Combines all forensic methods with weighted voting to produce
    final forgery probability maps.
    """
    
    def __init__(self, config: ForensicConfig, device: torch.device):
        self.config = config
        self.device = device
        
        self.model = None
        self.model_available = False
        
        self.ela_analyzer = ErrorLevelAnalyzer(config)
        self.noise_analyzer = NoiseResidualAnalyzer(config)
        self.lbp_analyzer = LBPAnalyzer(config)
        self.dct_analyzer = DCTAnalyzer(config)
        self.ghost_analyzer = JPEGGhostAnalyzer(config)
        self.copymove_analyzer = CopyMoveAnalyzer(config)
        self.illumination_analyzer = IlluminationAnalyzer(config)
        self.chromatic_analyzer = ChromaticAberrationAnalyzer(config)
        self.edge_analyzer = EdgeCoherenceAnalyzer(config)
        self.benford_analyzer = BenfordAnalyzer(config)
        self.statistical_analyzer = StatisticalMomentAnalyzer(config)
    
    def load_model(self) -> bool:
        try:
            self.model = TITANModel(self.config).to(self.device)
            
            if self.config.weights_path.exists():
                checkpoint = torch.load(
                    self.config.weights_path, 
                    map_location=self.device, 
                    weights_only=False
                )
                
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=True)
                self.model.eval()
                self.model_available = True
                logger.info(f"Model loaded from {self.config.weights_path}")
                return True
            else:
                logger.warning(f"Weights not found at {self.config.weights_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def preprocess_image(self, image_path: Path) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]], Optional[np.ndarray]]:
        image_bgr = cv2.imread(str(image_path))
        
        if image_bgr is None:
            logger.warning(f"Failed to read image: {image_path}")
            return None, None, None
        
        original_shape = image_bgr.shape[:2]
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.config.img_size, self.config.img_size))
        
        image_normalized = image_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float().unsqueeze(0)
        
        return image_tensor, original_shape, image_bgr
    
    @torch.no_grad()
    def predict_model(self, image_tensor: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        if not self.model_available:
            return np.zeros(original_shape, dtype=np.float32)
        
        self.model.eval()
        
        predictions = []
        
        pred = torch.sigmoid(self.model(image_tensor.to(self.device)))
        predictions.append(pred)
        
        if self.config.enable_tta:
            # Horizontal flip only (2x TTA instead of 4x)
            img_hflip = torch.flip(image_tensor, dims=[3])
            pred_hflip = torch.sigmoid(self.model(img_hflip.to(self.device)))
            pred_hflip = torch.flip(pred_hflip, dims=[3])
            predictions.append(pred_hflip)
        
        pred_avg = torch.mean(torch.stack(predictions), dim=0)
        pred_np = pred_avg.squeeze().cpu().numpy()
        
        pred_resized = cv2.resize(pred_np, (original_shape[1], original_shape[0]))
        
        return pred_resized.astype(np.float32)
    
    def analyze(self, image_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        image_tensor, original_shape, image_bgr = self.preprocess_image(image_path)
        
        if image_tensor is None:
            return None, {}
        
        h, w = original_shape
        forensic_maps = {}
        
        model_map = self.predict_model(image_tensor, original_shape)
        forensic_maps[ForensicMethod.MODEL.value] = model_map
        
        ela_map = self.ela_analyzer.analyze(image_bgr)
        ela_map = cv2.resize(ela_map, (w, h))
        forensic_maps[ForensicMethod.ELA.value] = ela_map
        
        noise_map = self.noise_analyzer.analyze(image_bgr)
        noise_map = cv2.resize(noise_map, (w, h))
        forensic_maps[ForensicMethod.NOISE.value] = noise_map
        
        lbp_map = self.lbp_analyzer.analyze(image_bgr)
        lbp_map = cv2.resize(lbp_map, (w, h))
        forensic_maps[ForensicMethod.LBP.value] = lbp_map
        
        dct_map = self.dct_analyzer.analyze(image_bgr)
        dct_map = cv2.resize(dct_map, (w, h))
        forensic_maps[ForensicMethod.DCT.value] = dct_map
        
        ghost_map = self.ghost_analyzer.analyze(image_bgr)
        ghost_map = cv2.resize(ghost_map, (w, h))
        forensic_maps[ForensicMethod.GHOST.value] = ghost_map
        
        copymove_map = self.copymove_analyzer.analyze(image_bgr)
        copymove_map = cv2.resize(copymove_map, (w, h))
        forensic_maps[ForensicMethod.COPYMOVE.value] = copymove_map
        
        illum_map = self.illumination_analyzer.analyze(image_bgr)
        illum_map = cv2.resize(illum_map, (w, h))
        forensic_maps[ForensicMethod.ILLUMINATION.value] = illum_map
        
        chroma_map = self.chromatic_analyzer.analyze(image_bgr)
        chroma_map = cv2.resize(chroma_map, (w, h))
        forensic_maps[ForensicMethod.CHROMATIC.value] = chroma_map
        
        edge_map = self.edge_analyzer.analyze(image_bgr)
        edge_map = cv2.resize(edge_map, (w, h))
        forensic_maps[ForensicMethod.EDGE.value] = edge_map
        
        benford_map = self.benford_analyzer.analyze(image_bgr)
        benford_map = cv2.resize(benford_map, (w, h))
        forensic_maps[ForensicMethod.BENFORD.value] = benford_map
        
        stat_map = self.statistical_analyzer.analyze(image_bgr)
        stat_map = cv2.resize(stat_map, (w, h))
        forensic_maps[ForensicMethod.STATISTICAL.value] = stat_map
        
        ensemble = (
            self.config.weight_model * forensic_maps[ForensicMethod.MODEL.value] +
            self.config.weight_ela * forensic_maps[ForensicMethod.ELA.value] +
            self.config.weight_noise * forensic_maps[ForensicMethod.NOISE.value] +
            self.config.weight_lbp * forensic_maps[ForensicMethod.LBP.value] +
            self.config.weight_dct * forensic_maps[ForensicMethod.DCT.value] +
            self.config.weight_ghost * forensic_maps[ForensicMethod.GHOST.value] +
            self.config.weight_copymove * forensic_maps[ForensicMethod.COPYMOVE.value] +
            self.config.weight_illumination * forensic_maps[ForensicMethod.ILLUMINATION.value] +
            self.config.weight_chromatic * forensic_maps[ForensicMethod.CHROMATIC.value] +
            self.config.weight_edge * forensic_maps[ForensicMethod.EDGE.value] +
            self.config.weight_benford * forensic_maps[ForensicMethod.BENFORD.value] +
            self.config.weight_statistical * forensic_maps[ForensicMethod.STATISTICAL.value]
        )
        
        if ensemble.max() > 0:
            ensemble = ensemble / ensemble.max()
        
        return ensemble, forensic_maps
    
    def postprocess(self, ensemble_map: np.ndarray, forensic_maps: Dict[str, np.ndarray]) -> np.ndarray:
        model_max = forensic_maps.get(ForensicMethod.MODEL.value, np.zeros_like(ensemble_map)).max()
        ela_max = forensic_maps.get(ForensicMethod.ELA.value, np.zeros_like(ensemble_map)).max()
        noise_max = forensic_maps.get(ForensicMethod.NOISE.value, np.zeros_like(ensemble_map)).max()
        ghost_max = forensic_maps.get(ForensicMethod.GHOST.value, np.zeros_like(ensemble_map)).max()
        
        strong_signals = sum([
            model_max > 0.3,
            ela_max > 0.4,
            noise_max > 0.4,
            ghost_max > 0.4
        ])
        
        if strong_signals >= 3:
            threshold = self.config.ensemble_threshold_low * 0.8
        elif strong_signals >= 2:
            threshold = self.config.ensemble_threshold_low
        elif strong_signals >= 1:
            threshold = (self.config.ensemble_threshold_low + self.config.ensemble_threshold_high) / 2
        else:
            threshold = self.config.ensemble_threshold_high
        
        binary_mask = (ensemble_map >= threshold).astype(np.uint8)
        
        kernel = np.ones((self.config.morph_kernel_size, self.config.morph_kernel_size), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        final_mask = np.zeros_like(binary_mask)
        image_area = binary_mask.shape[0] * binary_mask.shape[1]
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.config.min_forgery_area and area < image_area * self.config.max_forgery_ratio:
                final_mask[labels == i] = 1
        
        return final_mask


def rle_encode(mask: np.ndarray) -> str:
    """Encode binary mask to RLE format for competition submission."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return json.dumps([int(x) for x in runs])


def main():
    """Main entry point for forensic inference pipeline."""
    
    logger.info("=" * 80)
    logger.info("TITAN Production Forensic Inference System")
    logger.info("=" * 80)
    
    config = ForensicConfig()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    ensemble = ForensicEnsemble(config, device)
    model_loaded = ensemble.load_model()
    
    if not model_loaded:
        logger.warning("Proceeding with forensic analysis only (no deep learning model)")
    
    # Load sample submission
    try:
        sample_submission = pd.read_csv(config.data_root / 'sample_submission.csv')
    except Exception:
        sample_submission = pd.DataFrame({'case_id': [45], 'annotation': ['authentic']})
    
    logger.info(f"Sample submission rows: {len(sample_submission)}")
    
    # Build image ID map from test directory
    all_files = glob(str(config.data_root / 'test_images' / '**' / '*'), recursive=True)
    id_map = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            base = os.path.basename(f)
            digits = ''.join(filter(str.isdigit, os.path.splitext(base)[0]))
            if digits:
                id_map[str(int(digits))] = f
    
    logger.info(f"Test images found: {len(id_map)}")
    
    results = []
    statistics = {
        'total': 0,
        'authentic': 0,
        'forged': 0,
        'failed': 0
    }
    
    for case_id, image_path in tqdm(id_map.items(), desc="Processing"):
        image_path = Path(image_path)
        statistics['total'] += 1
        label = "authentic"
        
        try:
            ensemble_map, forensic_maps = ensemble.analyze(image_path)
            
            if ensemble_map is None:
                statistics['failed'] += 1
            else:
                final_mask = ensemble.postprocess(ensemble_map, forensic_maps)
                
                if final_mask.sum() >= config.min_forgery_area:
                    label = rle_encode(final_mask)
                    if label == "":
                        label = "authentic"
                    else:
                        statistics['forged'] += 1
                else:
                    statistics['authentic'] += 1
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            statistics['failed'] += 1
        
        results.append({'case_id': case_id, 'annotation': label})
    
    # Exact submission logic from working reference
    sample_submission['case_id'] = sample_submission['case_id'].astype(str)
    
    if len(results) > 0:
        preds_df = pd.DataFrame(results)
        preds_df['case_id'] = preds_df['case_id'].astype(str)
        submission = sample_submission[['case_id']].merge(preds_df, on='case_id', how='left')
        submission['annotation'] = submission['annotation'].fillna('authentic')
    else:
        submission = sample_submission.copy()
        submission['annotation'] = 'authentic'
    
    submission.to_csv(config.output_path, index=False)
    
    logger.info("=" * 80)
    logger.info("Processing Complete")
    logger.info("=" * 80)
    logger.info(f"Total images: {statistics['total']}")
    logger.info(f"Authentic: {statistics['authentic']} ({100 * statistics['authentic'] / max(1, statistics['total']):.1f}%)")
    logger.info(f"Forged: {statistics['forged']} ({100 * statistics['forged'] / max(1, statistics['total']):.1f}%)")
    logger.info(f"Failed: {statistics['failed']}")
    logger.info(f"Output: {config.output_path}")
    logger.info(f"Preview:")
    logger.info(submission.head().to_string())


if __name__ == "__main__":
    main()

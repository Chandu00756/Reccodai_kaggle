
# TITAN: Scientific Image Forgery Detection

Welcome to the TITAN project, a complete solution for detecting image forgeries in scientific datasets. This repository brings together advanced deep learning and classical forensic analysis to help researchers, data scientists, and engineers identify manipulated or tampered images with confidence.

---

## Overview

TITAN is designed to tackle the challenge of scientific image forgery by combining state-of-the-art neural networks with proven forensic techniques. The system is modular, robust, and easy to use for both research and real-world applications.

**Key Features:**
- Swin Transformer + UNet-based segmentation model
- Ensemble of forensic methods: Error Level Analysis, Noise Residuals, LBP, DCT, JPEG Ghost, Copy-Move, Illumination, Chromatic Aberration, Edge Coherence, Benford's Law, Statistical Moments
- Flexible training and inference pipelines
- Jupyter notebooks for exploration and validation
- Clean, organized project structure for easy collaboration

---

## Folder Structure

```
src/                # All Python scripts (training, inference, utilities)
notebooks/          # Jupyter notebooks for experiments and analysis
models/             # Model weights (.pth files)
data/               # Local data, small samples, or CSVs
checkpoints/        # Training checkpoints
old_weights/        # Legacy model weights
model_audit/        # Model audit scripts and results
recodai-luc-scientific-image-forgery-detection/  # Main dataset and supplemental files
    supplemental_images/
    supplemental_masks/
    test_images/
    train_images/
    train_masks/
utils/              # Helper scripts (if any)
requirements.txt    # Python dependencies
.gitignore          # Git ignore rules
README.md           # Project documentation
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Chandu00756/Reccodai_kaggle.git
cd Reccodai_kaggle
```

### 2. Install Dependencies

Install Python 3.8+ and all required packages:

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place your images and masks in the appropriate folders under `recodai-luc-scientific-image-forgery-detection/`.

### 4. Training

To train the TITAN model on your data:

```bash
python src/TITAN_TRAIN_08.py
```

Training checkpoints and logs will be saved in the `checkpoints/` directory.

### 5. Inference

To run inference and generate forgery predictions:

```bash
python src/TITAN_PRODUCTION_INFERENCE.py
```

You can also use other scripts in `src/` for fast or alternative inference modes.

### 6. Notebooks

Explore the `notebooks/` directory for Jupyter notebooks that demonstrate model training, validation, and forensic analysis. These are ideal for experimentation and visualization.

---

## Forensic Methods

TITAN leverages a diverse set of forensic techniques, including:
- Deep learning segmentation (Swin-UNet)
- Error Level Analysis (ELA)
- Noise Residual Analysis
- Local Binary Patterns (LBP)
- DCT Coefficient Analysis
- JPEG Ghost Detection
- Copy-Move Detection
- Illumination Consistency
- Chromatic Aberration
- Edge Coherence
- Benford's Law
- Statistical Moments

These methods are combined in an ensemble to maximize detection accuracy and robustness.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have improvements, bug fixes, or new ideas.

---

## License

This project is released for research and practical use. See individual source files for detailed license information.

---

## Contact

For questions, support, or collaboration, please contact the repository maintainer via GitHub Issues or Discussions.

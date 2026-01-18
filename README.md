# Scientific Image Forgery Detection (TITAN)

This repository contains the TITAN system for scientific image forgery detection, including deep learning and classical forensic methods.

## Project Structure

- `src/` — All Python scripts (training, inference, utilities)
- `notebooks/` — Jupyter notebooks for experiments and analysis
- `models/` — Model weights (.pth files)
- `data/` — For local data, small samples, or CSVs
- `checkpoints/` — Training checkpoints
- `old_weights/` — Legacy model weights
- `model_audit/` — Model audit scripts and results
- `recodai-luc-scientific-image-forgery-detection/` — Main dataset and supplemental files
    - `supplemental_images/`, `supplemental_masks/`, `test_images/`, `train_images/`, `train_masks/`
- `utils/` — Helper scripts (if any)

## Main Scripts
- Training: `src/TITAN_TRAIN_08.py`
- Inference: `src/TITAN_PRODUCTION_INFERENCE.py`, `src/TITAN_FAST_FORENSIC.py`, etc.

## Usage
1. Place your data in the appropriate folders.
2. Run training or inference scripts from `src/`.
3. Notebooks for analysis are in `notebooks/`.

## Requirements
See `requirements.txt` for dependencies.

## License
Production use permitted. See source files for details.

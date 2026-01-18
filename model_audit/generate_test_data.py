import os
import cv2
import random
import numpy as np
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your AUTHENTIC training images (source material)
# Update this to match your local folder structure!
SOURCE_DIR = "/Users/chanduchitikam/recodai/recodai-luc-scientific-image-forgery-detection/train_images/authentic" 

# Where to save the new test set (will save in model_audit folder)
OUTPUT_IMG_DIR = os.path.join(SCRIPT_DIR, "images")
OUTPUT_MASK_DIR = os.path.join(SCRIPT_DIR, "masks")

NUM_IMAGES_TO_GENERATE = 200
MIN_OBJECT_SIZE = 40
MAX_OBJECT_SIZE = 100

def create_synthetic_forgery(src_path, filename_idx):
    img = cv2.imread(src_path)
    if img is None: return
    
    h, w, c = img.shape
    
    # SAFETY CHECK 1: If image is too small, skip it
    if h < MIN_OBJECT_SIZE or w < MIN_OBJECT_SIZE:
        return

    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 1. Select a random patch (The "Source")
    for _ in range(10):
        # SAFETY CHECK 2: Ensure patch is never bigger than the image
        safe_max_h = min(MAX_OBJECT_SIZE, h - 1)
        safe_max_w = min(MAX_OBJECT_SIZE, w - 1)
        
        # If image is barely bigger than MIN_SIZE, force min size
        if safe_max_h < MIN_OBJECT_SIZE: safe_max_h = MIN_OBJECT_SIZE
        if safe_max_w < MIN_OBJECT_SIZE: safe_max_w = MIN_OBJECT_SIZE
        
        ph = random.randint(MIN_OBJECT_SIZE, safe_max_h)
        pw = random.randint(MIN_OBJECT_SIZE, safe_max_w)
        
        y = random.randint(0, h - ph)
        x = random.randint(0, w - pw)
        
        patch = img[y:y+ph, x:x+pw]
        
        # Simple check: Is this patch interesting? (Not just solid black)
        if np.std(patch) > 10: 
            break
    else:
        return # Skip if we couldn't find a good patch
        
    # 2. Augment the Patch (Test your TTA!)
    if random.random() > 0.5:
        patch = cv2.flip(patch, 1)
        
    k = random.randint(0, 3)
    if k > 0:
        patch = np.rot90(patch, k)
        
    # 3. Paste the Patch (The "Target")
    ph_new, pw_new, _ = patch.shape
    
    # SAFETY CHECK 3: If rotation made patch bigger than image, skip
    if ph_new >= h or pw_new >= w:
        return
    
    # Find a target location
    for _ in range(10):
        ty = random.randint(0, h - ph_new)
        tx = random.randint(0, w - pw_new)
        
        # Check simple overlap
        if abs(ty - y) > ph or abs(tx - x) > pw:
            break
            
    img[ty:ty+ph_new, tx:tx+pw_new] = patch
    mask[ty:ty+ph_new, tx:tx+pw_new] = 255
    
    # Save
    out_name = f"synth_test_{filename_idx:03d}"
    cv2.imwrite(f"{OUTPUT_IMG_DIR}/{out_name}.png", img)
    cv2.imwrite(f"{OUTPUT_MASK_DIR}/{out_name}.png", mask)

def main():
    # Setup folders
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    
    # Get source images
    authentic_files = glob.glob(f"{SOURCE_DIR}/*.png")
    if len(authentic_files) == 0:
        print(f"ERROR: No source images found in {SOURCE_DIR}")
        return
        
    print(f">>> Generating {NUM_IMAGES_TO_GENERATE} Synthetic Forgeries...")
    
    # Randomly sample images
    # Ensure we don't try to sample more than exist
    sample_count = min(len(authentic_files), NUM_IMAGES_TO_GENERATE)
    selected_files = random.sample(authentic_files, sample_count)
    
    for i, p in enumerate(tqdm(selected_files)):
        create_synthetic_forgery(p, i)
        
    print("\n>>> Generation Complete.")
    print(f"    Images saved to: {OUTPUT_IMG_DIR}")
    print(f"    Masks saved to:  {OUTPUT_MASK_DIR}")

if __name__ == "__main__":
    main()
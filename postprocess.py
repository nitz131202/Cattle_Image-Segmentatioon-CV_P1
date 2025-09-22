
import os
import cv2
import argparse
import numpy as np

def clean_mask(mask):
    """Apply postprocessing to clean a binary mask."""
    # Convert to binary
    mask = (mask > 127).astype(np.uint8) * 255

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove noise

    # Keep largest connected component (the cow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, [largest], -1, 255, thickness=cv2.FILLED)
        mask = new_mask

    return mask

def process_masks(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    for f in files:
        path = os.path.join(input_folder, f)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        cleaned = clean_mask(mask)

        out_path = os.path.join(output_folder, f)
        cv2.imwrite(out_path, cleaned)
        print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root segmentation_output folder")
    args = parser.parse_args()

    # Walk through all cow/date directories
    for root, dirs, files in os.walk(args.root):
        if root.endswith("png/masks"):
            out_dir = root + "_post"
            print(f"Processing {root} -> {out_dir}")
            process_masks(root, out_dir)

if __name__ == "__main__":
    main()

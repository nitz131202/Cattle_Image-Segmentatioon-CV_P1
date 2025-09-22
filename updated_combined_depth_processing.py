
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import logging
from timeit import default_timer as timer
import shutil
import yaml
import warnings
import urllib.request
import ssl

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

# Add the parent directory of 'unet' to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import your custom modules from the unet/src directory
from unet.src.data_lib import MyCustomDataset, DataLoader
from unet.src.train_lib import TrainingModel, Trainer, TRAIN_MODEL_NAME
from unet.src.utils import get_device, draw_mask_over_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check Albumentations version and warn if outdated
try:
    import albumentations
    current_version = albumentations.__version__
    if current_version < "1.4.14":
        logging.warning(f"A new version of Albumentations is available: 1.4.14 (you have {current_version}). "
                        "Consider upgrading using: pip install --upgrade albumentations")
except ImportError:
    logging.warning("Albumentations not found. Make sure it's installed.")

# Function to download ResNet34 weights
def download_resnet34_weights():
    url = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
    filename = os.path.join(os.getcwd(), "resnet34-333f7ec4.pth")
    if not os.path.exists(filename):
        logging.info("Downloading ResNet34 weights...")
        # Ignore SSL certificate errors
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, filename)
        logging.info("ResNet34 weights downloaded successfully.")
    return filename

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.config_path = config_path

    def get(self, key):
        keys = key.split('/')
        value = self.config
        for k in keys:
            value = value.get(k)
            if value is None:
                return None
        return value

def convert_to_grayscale(input_folder, output_folder):
    """Convert depth TIFF images to grayscale using robust normalization."""
    if not os.path.exists(input_folder):
        logging.error(f"Input folder not found: {input_folder}")
        return False

    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
    if not files:
        logging.error(f"No TIFF files found in {input_folder}")
        return False

    os.makedirs(output_folder, exist_ok=True)

    for filename in files:
        try:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

            # Read TIFF image as grayscale float32
            depth_data = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

            if depth_data is None or depth_data.size == 0:
                logging.warning(f"Failed to read image or empty: {filename}")
                continue

            # Robust normalization
            lower_bound = np.percentile(depth_data, 1)
            upper_bound = np.percentile(depth_data, 99)
            clipped_data = np.clip(depth_data, lower_bound, upper_bound)
            normalized = ((clipped_data - lower_bound) * 255 / (upper_bound - lower_bound)).astype(np.uint8)

            success = cv2.imwrite(output_path, normalized)
            if not success:
                logging.error(f"Failed to save image: {output_path}")

        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")

    converted_count = len([f for f in os.listdir(output_folder) if f.endswith('.png')])
    logging.info(f"Successfully converted {converted_count} out of {len(files)} TIFF images to grayscale")
    return converted_count > 0


def process_and_infer(grayscale_folder, output_folder, cfg, model_path):
    """Process grayscale images and run inference."""
    if not os.path.exists(model_path) or not os.listdir(grayscale_folder):
        return False

    try:
        device = get_device()
        model = TrainingModel(cfg.get('model'), device=device)
        model.model.load_state_dict(torch.load(model_path, map_location=device))
        model.model.to(device)

        optimizer = TrainingModel(cfg.get('optimizer'), opt=model.model.parameters())
        scheduler = TrainingModel(cfg.get('scheduler'), opt=optimizer.model)
        criterion = TrainingModel(cfg.get('criterion'))

        img_paths = [os.path.join(grayscale_folder, f) for f in os.listdir(grayscale_folder) if f.endswith('.png')]
        dataset = MyCustomDataset(cfg.get('augmentation'), img_paths, with_mask=False)
        dataloader = DataLoader(batchsize=16, pin_memory=True, shuffle=False)
        dataloader = dataloader(dataset)

        masks_dir = os.path.join(output_folder, 'png', 'masks')
        splash_dir = os.path.join(output_folder, 'png', 'splash')
        
        trainer = Trainer(model, optimizer, criterion, scheduler)
        trainer.inference_save_direct(dataloader, masks_dir, splash_dir)
        
        return True

    except Exception as e:
        logging.error(f"Error in process_and_infer: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process depth numpy arrays and run inference.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing tiff depth files")
    parser.add_argument("--output_dir", default="output", help="Path to the output directory")
    parser.add_argument("--config_path", default="unet/config/config.yml", help="Path to the configuration file")
    parser.add_argument("--model_path", default="unet/models/cows_scale_100.pth", help="Path to the trained model")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )

    # Verify input paths
    for path in [args.input_folder, args.config_path, args.model_path]:
        if not os.path.exists(path):
            logging.error(f"Path not found: {path}")
            sys.exit(1)

    try:
        cfg = Config(args.config_path)
    except Exception as e:
        logging.error(f"Failed to load config: {str(e)}")
        sys.exit(1)

    start_time = timer()

    # Set up clean output structure with only npy and png directories
    output_folder = args.output_dir
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # Create clean directory structure
    for subdir in ['png/grayscale', 'png/masks', 'png/splash']:
        os.makedirs(os.path.join(output_folder, subdir))

    # Process depth files and convert to grayscale
    depth_folder = os.path.join(args.input_folder)  # ← now points directly to TIFF images
    grayscale_folder = os.path.join(output_folder, 'png', 'grayscale')
    
    if not convert_to_grayscale(depth_folder, grayscale_folder):
        sys.exit(1)

    # Process and infer
    if not process_and_infer(grayscale_folder, output_folder, cfg, args.model_path):
        sys.exit(1)

    # Copy IR PNGs from sibling folder (IR is alongside Depth, not inside)
    cow_dir = os.path.dirname(args.input_folder)  # go one level up from Depth
    src = os.path.join(cow_dir, 'IR')
    dst = os.path.join(output_folder, 'png', 'infra')  # keep in PNG instead of npy
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logging.info(f"Copied IR PNG folder to {dst}")
    else:
        logging.warning(f"IR folder not found at {src}")

    end_time = timer()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

import os
import subprocess
from datetime import datetime

# Set your base paths
base_input_dir = "/Users/sreenithyam/Desktop/Project 1/extracted_frames"
base_output_dir = "/Users/sreenithyam/Desktop/Project 1/segmentation_output"
config_path = "/Users/sreenithyam/Desktop/Project 1/segmentation/unet/config/config.yml"
model_path = "/Users/sreenithyam/Desktop/Project 1/marshfield_chute_100.pth"

# Loop through all date folders
for date in sorted(os.listdir(base_input_dir)):
    date_path = os.path.join(base_input_dir, date)
    if not os.path.isdir(date_path):
        continue

    # Loop through cow folders
    for cow_id in sorted(os.listdir(date_path)):
        cow_path = os.path.join(date_path, cow_id)
        depth_path = os.path.join(cow_path, "Depth")

        if not os.path.isdir(depth_path):
            print(f"⚠️ Skipping {cow_id}: Depth folder not found.")
            continue

        print(f"Processing {cow_id} on {date}...")

        output_dir = os.path.join(base_output_dir, date, cow_id)

        # Construct the command to run your main script
        cmd = [
            "python3", "segmentation_script.py",
            "--input_folder", depth_path,
            "--output_dir", output_dir,
            "--config_path", config_path,
            "--model_path", model_path
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to process {cow_id} on {date}: {e}")
        else:
            print(f"Finished {cow_id} on {date}\n")


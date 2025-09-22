
import os
import subprocess

# Paths
base_input_dir = "/Users/sreenithyam/Desktop/Project 1/extracted_frames"
base_output_dir = "/Users/sreenithyam/Desktop/Project 1/segmentation_output"
config_path = "/Users/sreenithyam/Desktop/Project 1/segmentation/unet/config/config.yml"
model_path = "/Users/sreenithyam/Desktop/Project 1/segmentation/marshfield_chute_100.pth"

# Path to your updated single-folder script
script_path = "/Users/sreenithyam/Desktop/Project 1/segmentation/updated_combined_depth_processing.py"

for date in sorted(os.listdir(base_input_dir)):
    date_path = os.path.join(base_input_dir, date)
    if not os.path.isdir(date_path):
        continue

    for cow_id in sorted(os.listdir(date_path)):
        depth_path = os.path.join(date_path, cow_id, "Depth")
        if not os.path.isdir(depth_path):
            print(f"⚠️ Skipping {cow_id} on {date} (no Depth folder)")
            continue

        output_dir = os.path.join(base_output_dir, date, cow_id)
        print(f"🚀 Processing {cow_id} on {date}...")
        print(f"   Input: {depth_path}")
        print(f"   Output: {output_dir}")

        cmd = [
            "python3", script_path,
            "--input_folder", depth_path,
            "--output_dir", output_dir,
            "--config_path", config_path,
            "--model_path", model_path,
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed {cow_id} on {date}: {e}")
        else:
            print(f"✅ Finished {cow_id} on {date}\n")

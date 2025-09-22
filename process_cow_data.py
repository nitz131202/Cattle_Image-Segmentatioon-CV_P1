import os
import csv
import shutil
import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime, timedelta
from tqdm import tqdm

# === USER CONFIG ===
source_dir = r'Z:\3DScale'  # NAS path
calving_csv = r'C:\Users\Patron\Desktop\Project\cow_calving_guessed.csv'  # Updated path
local_copy_dir = r'C:\FilteredBags'  # Folder for copied .bag files
output_dir = r'C:\ExtractedFrames'   # Folder for extracted frames

os.makedirs(local_copy_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

frame_interval = 1  # Extract 1 frame every second
frame_rate = 30
frames_to_skip = frame_rate * frame_interval

# === STEP 1: Load calving dates and generate offsets ===
calving_dates = {}
with open(calving_csv, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row[1].strip():
            calving_dates[row[0].strip()] = datetime.strptime(row[1].strip(), "%Y%m%d")

offsets = [-21, -14, -7]
target_dates = {cow: [(date + timedelta(days=o)).strftime("%Y%m%d") for o in offsets]
                for cow, date in calving_dates.items()}

# === STEP 2: Find and copy matched files ===
matched_files = []
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        if file.endswith('.bag'):
            parts = file.split('_')
            if len(parts) >= 3:
                cow_id, file_date = parts[0], parts[1]
                if cow_id in target_dates and file_date in target_dates[cow_id]:
                    src = os.path.join(folder_path, file)
                    dst = os.path.join(local_copy_dir, file)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    matched_files.append(dst)

print(f"✅ Copied {len(matched_files)} files to {local_copy_dir}")

# === STEP 3: Extract frames ===
summary = [["File", "Frames Saved", "RGB", "Depth", "IR"]]

for video_path in tqdm(matched_files, desc="Extracting Frames"):
    video_file = os.path.basename(video_path)
    save_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
    rgb_dir = os.path.join(save_dir, "RGB")
    depth_dir = os.path.join(save_dir, "Depth")
    ir_dir = os.path.join(save_dir, "IR")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    try:
        rs.config.enable_device_from_file(config, video_path, repeat_playback=False)
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        frame_count = saved_frame_count = 0
        rgb_saved = depth_saved = ir_saved = 0

        while True:
            success, frames = pipeline.try_wait_for_frames(100)
            if not success:
                break

            if frame_count % frames_to_skip == 0:
                # Depth
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(depth_dir, f"depth_{saved_frame_count}.tif"), depth_image)
                    cv2.imwrite(os.path.join(depth_dir, f"depth_vis_{saved_frame_count}.png"), depth_colormap)
                    depth_saved += 1

                # RGB
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(rgb_dir, f"rgb_{saved_frame_count}.png"), bgr)
                    rgb_saved += 1

                # IR
                ir_frame = frames.get_infrared_frame()
                if ir_frame:
                    ir_image = np.asanyarray(ir_frame.get_data())
                    ir_norm = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
                    cv2.imwrite(os.path.join(ir_dir, f"ir_{saved_frame_count}.png"), ir_norm)
                    ir_saved += 1

                saved_frame_count += 1

            frame_count += 1

        summary.append([video_file, saved_frame_count, rgb_saved, depth_saved, ir_saved])
    except Exception as e:
        print(f"❌ Error processing {video_file}: {e}")
    finally:
        pipeline.stop()

# Save summary CSV
summary_csv = os.path.join(output_dir, "extraction_summary.csv")
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(summary)

print(f"\n✅ All files processed. Summary saved at {summary_csv}")



import os
import pyrealsense2 as rs
import numpy as np
import cv2
from tqdm import tqdm
import csv

# === CONFIG ===
input_dir = r"C:\FilteredBags"
output_dir = r"C:\ExtractedFrames"
os.makedirs(output_dir, exist_ok=True)

frame_interval = 3  # Save 1 frame every 3 seconds
frame_rate = 30
frames_to_skip = frame_rate * frame_interval

extract_rgb = True
extract_depth = True
extract_ir = False  # Set to True if you need IR

# Get all bag files
bag_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".bag")]

print(f"Found {len(bag_files)} .bag files to process.")
summary = [["File", "Cow_ID", "Date", "Frames Saved", "RGB", "Depth", "IR"]]

for video_path in tqdm(bag_files, desc="Extracting Frames"):
    video_file = os.path.basename(video_path)
    cow_id, date_str, _ = video_file.split("_")
    save_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
    os.makedirs(save_dir, exist_ok=True)

    rgb_dir = os.path.join(save_dir, "RGB")
    depth_dir = os.path.join(save_dir, "Depth")
    ir_dir = os.path.join(save_dir, "IR")

    if extract_rgb: os.makedirs(rgb_dir, exist_ok=True)
    if extract_depth: os.makedirs(depth_dir, exist_ok=True)
    if extract_ir: os.makedirs(ir_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    started = False  # Track if pipeline started

    try:
        rs.config.enable_device_from_file(config, video_path, repeat_playback=False)
        profile = pipeline.start(config)
        started = True
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        frame_count = saved_frame_count = 0
        rgb_saved = depth_saved = ir_saved = 0

        while True:
            success, frames = pipeline.try_wait_for_frames(100)
            if not success:
                break

            if frame_count % frames_to_skip == 0:
                if extract_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_image = np.asanyarray(depth_frame.get_data())
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                        cv2.imwrite(os.path.join(depth_dir, f"depth_{saved_frame_count}.tif"), depth_image)
                        cv2.imwrite(os.path.join(depth_dir, f"depth_vis_{saved_frame_count}.png"), depth_colormap)
                        depth_saved += 1

                if extract_rgb:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_image = np.asanyarray(color_frame.get_data())
                        bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(rgb_dir, f"rgb_{saved_frame_count}.png"), bgr)
                        rgb_saved += 1

                if extract_ir:
                    ir_frame = frames.get_infrared_frame()
                    if ir_frame:
                        ir_image = np.asanyarray(ir_frame.get_data())
                        ir_norm = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX)
                        cv2.imwrite(os.path.join(ir_dir, f"ir_{saved_frame_count}.png"), ir_norm)
                        ir_saved += 1

                saved_frame_count += 1

            frame_count += 1

        summary.append([video_file, cow_id, date_str, saved_frame_count, rgb_saved, depth_saved, ir_saved])
        print(f"✔ Done {video_file} | Frames saved: {saved_frame_count}")

    except Exception as e:
        print(f"❌ Error processing {video_file}: {e}")
        summary.append([video_file, cow_id, date_str, "ERROR", 0, 0, 0])

    finally:
        if started:
            pipeline.stop()

# Save summary CSV
summary_csv = os.path.join(output_dir, "extraction_summary.csv")
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(summary)

print(f"\n✅ Extraction complete. Summary saved at {summary_csv}")

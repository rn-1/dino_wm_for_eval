import os
import shutil

import robomimic.utils.file_utils as FileUtils

download_folder = "/project2/jessetho_1732/rl_eval_wm/dino_wm/data"
os.makedirs(download_folder, exist_ok=True)

# Image URL follows the same pattern as low_dim but is not in the registry
url = "http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/image_v141.hdf5"
FileUtils.download_url(url=url, download_dir=download_folder)

# Rename to the expected filename
downloaded_name = os.path.join(download_folder, "image_v141.hdf5")
target_path = os.path.join(download_folder, "robomimic.hdf5")
if os.path.exists(downloaded_name):
    shutil.move(downloaded_name, target_path)
assert os.path.exists(target_path), f"Dataset not found at {target_path}"
print(f"Dataset saved to {target_path}")
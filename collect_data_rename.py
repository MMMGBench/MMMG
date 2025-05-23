import os
import json
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count

# ---------- Config ----------
base_dir = "/detr_blob/v-luoyuxuan/EVALUATE-IMG-GEN/results/WKBENCH/4o/all_prompts_refined_output"
dest_dir = "/detr_blob/v-luoyuxuan/MMMG_DATASET_HF/MMMG/data/GPT-4o/"
mapping_path = "/detr_blob/v-luoyuxuan/MMMG_DATASET_HF/old_key_to_new_key.json"
num_workers = max(cpu_count() - 1, 1)  # leave 1 core free
# -----------------------------

with open(mapping_path, "r", encoding="utf-8") as f:
    old_key_to_new_key = json.load(f)


def collect_png_paths(root: str):
    """Recursively collect all .png image paths under *root*."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".png"):
                yield os.path.join(dirpath, name)


def copy_one(img_path: str, mapping: dict, dest_root: str):
    """Copy *img_path* to the appropriate stage folder in *dest_root*."""
    old_key = os.path.basename(img_path).split(".")[0]  # e.g. "preschool_1_1"
    #print(f"Processing {old_key}...")
    new_key = mapping.get(old_key)
    if new_key is None:
        # No mapping found – skip
        return

    stage = new_key.split("_", 1)[0]  # e.g. preschool / primaryschool …
    stage_dir = os.path.join(dest_root, stage)
    os.makedirs(stage_dir, exist_ok=True)

    dest_path = os.path.join(stage_dir, new_key+".png")
    # If already copied, skip to avoid overwriting unnecessarily
    if os.path.exists(dest_path):
        return

    shutil.copyfile(img_path, dest_path)


def main():
    img_paths = list(collect_png_paths(base_dir))
    print(f"Found {len(img_paths)} images to copy.")
    worker = partial(copy_one, mapping=old_key_to_new_key, dest_root=dest_dir)

    # Multiprocess copy
    with Pool(processes=num_workers) as pool:
        pool.map(worker, img_paths)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Collect SAM-2.1 segmentation stats.

Example
-------
python collect_stats.py \
    --src_dir  /path/to/sam2_outputs \
    --dest_dir /path/to/statistics \
    --name     my_model \
    --num_workers 8
"""

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count


def process_file(src_dir: str, filename: str):
    """Read <src_dir>/<file>/anno.json and return a (key, info_dict) pair."""
    try:
        file_path = os.path.join(src_dir, filename, "anno.json")
        image_path = os.path.join(src_dir, filename, "sam2_result.png")

        with open(file_path, "r") as f:
            data = json.load(f)

        key = data["image_uid"].split("__")[-1].split(".")[0]
        return key, {
            "region_count": data["region_count"],
            "image_path":  data["image_path"],
            "sam_path":    image_path,
        }
    except Exception as e:  # noqa: BLE001
        print(f"[skip] {filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-image anno.json into a single stat file."
    )
    parser.add_argument("-s", "--src_dir",  required=True,
                        help="Root dir that contains per-image sub-folders")
    parser.add_argument("-o", "--dest_dir", required=True,
                        help="Folder to save the merged stat JSON")
    parser.add_argument("-n", "--name",
                        help="Prefix for output JSON (default: basename of src_dir)")
    parser.add_argument("-j", "--num_workers", type=int, default=cpu_count(),
                        help="Parallel workers (default: CPU cores)")
    args = parser.parse_args()

    src_dir  = os.path.abspath(args.src_dir)
    dest_dir = os.path.abspath(args.dest_dir)
    name     = args.name or os.path.basename(src_dir.rstrip("/"))
    os.makedirs(dest_dir, exist_ok=True)

    # enumerate sub-folders
    sub_dirs = [d for d in os.listdir(src_dir)
                if os.path.isdir(os.path.join(src_dir, d))]
    print(f"Found {len(sub_dirs)} samples in {src_dir}")

    # pool-map
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(partial(process_file, src_dir), sub_dirs)

    # flatten & filter None
    merged = {k: v for pair in results if pair for k, v in [pair]}

    # save
    out_path = os.path.join(dest_dir, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=4)
    print(f"Saved {len(merged)} entries → {out_path}")


if __name__ == "__main__":
    main()

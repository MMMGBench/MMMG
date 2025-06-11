import argparse
import json
import os
import re
from pathlib import Path
from all_configs import *
from PIL import Image 
from tqdm import tqdm

def clean_key(k):

    s = k.strip()
    s = re.sub(r"^[\*`]+", "", s)   
    s = re.sub(r"[\*`]+$", "", s)   
    s = re.sub(r"[:：]$", "", s)     
    return s.strip()

def parse_output_to_checklist(image_id, text, elements, dependencies):
    all_keys = elements + dependencies

    escaped_keys = [re.escape(k) for k in all_keys]

    text = text.replace("**", "")
    text = text.replace("[yes]", "yes")
    text = text.replace("[no]", "no")
    pattern = rf"^\s*(?:[*\-•]|\d+\.)?\s*`*\s*({'|'.join(escaped_keys)})`*\s*[:：]\s*(yes|no|[yes]|[no]|YES|NO|Yes|No)\b"  
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

    if len (matches) == 0:
        return {
            "key": image_id,
            "elements": {key: False for key in elements},
            "dependencies": {key: False for key in dependencies}
        }

    element_dict = {key: False for key in elements}
    dependency_dict = {key: False for key in dependencies}

    for raw_k, v in matches:
        k_clean = clean_key(raw_k)
        v_bool = v.strip().lower() == "yes"
        if k_clean in element_dict:
            element_dict[k_clean] = v_bool
        elif k_clean in dependency_dict:
            dependency_dict[k_clean] = v_bool
        else:
            find=False
            for ele in element_dict.keys():
                if k_clean.lower() == ele.lower():
                    element_dict[ele] = v_bool
                    find=True
                    break
            if not find:
                for ele in dependency_dict.keys():
                    if k_clean.lower() == ele.lower():
                        dependency_dict[ele] = v_bool
                        find=True
                        break
                

    return {
        "key": image_id,
        "elements": element_dict,
        "dependencies": dependency_dict
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge model answers with MMMG ground truth."
    )
    parser.add_argument("--result_folder", "-o", required=True,
                        help="Path containing model-generated *.json files")
    parser.add_argument("--image_folder", "-i", required=True,
                        help="Root folder holding six sub-folders of images")
    parser.add_argument("--api_name", "-a", required=True,
                        help="Key used in each result JSON to fetch the model output")
    parser.add_argument("--output_dir", required=True,
                        help="Folder to save merged result JSON")
    parser.add_argument("--save_name", default="step2_summarize",
                        help="Output file name (without .json)")
    parser.add_argument("--hf_cache", default="./data/MMMG",
                        help="HuggingFace cache dir")
    args = parser.parse_args()

    # ---------------- load MMMG ground truth ---------------------------------
    full_dataset = load_all_mmmg_configs(
        cache_dir=args.hf_cache, max_workers=16
    )

    # build lookup: {grade: {image_id: {elements, dependencies}}}
    gt = {}
    for sample in full_dataset:
        grade = str(sample["Education"])   # e.g. "preschool"
        image_id = sample["key"]
        kg = json.loads(sample["Knowledge_Graph"])
        gt.setdefault(grade, {})[image_id] = kg

    
    grade_map = {  # match folder names if they differ
        "preschool": "0_preschool",
        "primaryschool": "1_primaryschool",
        "secondaryschool": "2_secondaryschool",
        "highschool": "3_highschool",
        "undergraduate": "4_undergraduate",
        "PhD": "5_PhD",
    }

    merged = {g: {} for g in grade_map}

    # ---------------- scan model result files --------------------------------
    result_folder = Path(args.result_folder)
    image_folder = Path(args.image_folder)

    for fn in tqdm(sorted(result_folder.glob("*.json")), desc="merging"):
        try:
            data = json.loads(fn.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] cannot parse {fn.name}: {e}")
            continue

        if args.api_name not in data:
            print(f"[skip] {fn.name}: no key '{args.api_name}'")
            continue

        text = data[args.api_name]        # LLM checklist
        image_id = data["key"]              # assert match later

        parts = fn.stem.split("__")         # <grade>__<image_id>.json
        if len(parts) == 2:
            grade, img_key = parts
        else:
            grade, img_key,= parts[:2]

        if img_key != image_id:
            print(f"[warn] ID mismatch in {fn.name}")

        # locate image (png/jpg fallback)
        base = image_folder / grade
        if not base.exists():
            base = image_folder / grade_map.get(grade, grade)
        png_path = base / f"{image_id}.png"
        jpg_path = base / f"{image_id}.jpg"
        img_path = png_path if png_path.exists() else jpg_path
        if not img_path.exists():
            print(f"[skip] image not found for {image_id}")
            continue

        # ground-truth KG
        try:
            gt_entry = gt[grade][image_id]
            elements = gt_entry["elements"]
            dependencies = gt_entry["dependencies"]
        except KeyError:
            print(f"[skip] GT not found for {grade}/{image_id}")
            continue

        merged.setdefault(grade, {})[image_id] = {
            "img_path": str(img_path),
            "result": parse_output_to_checklist(image_id, text, elements, dependencies),
        }

    # ---------------- save ----------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.save_name}.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"Stage1 Evaluation Done ✔️. Saved to {out_path}")


if __name__ == "__main__":
    main()
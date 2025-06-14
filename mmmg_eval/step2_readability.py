import os, sys
import json
import argparse
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import base64
import io
import numpy as np
import torch
import torch.multiprocessing as mp
import os, sys, pathlib

PROJ_ROOT = pathlib.Path(__file__).resolve().parent 
SAM2_ROOT = PROJ_ROOT / "sam2"  

sys.path.insert(0, str(SAM2_ROOT))
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from paddleocr import PaddleOCR

import warnings
warnings.filterwarnings("ignore")


def call_ppocr(image_pil, image_width, image_height,
               confidence_threshold=0.85,
               box_short_edge_thresh=20,
               line_merge_y_thresh=20):
    """
    Perform OCR using PaddleOCR and return merged line-level text boxes in (x1, y1, x2, y2) format.

    Args:
        image_pil (PIL.Image): The image to perform OCR on.
        image_width (int): Image width.
        image_height (int): Image height.
        confidence_threshold (float): Minimum confidence to accept OCR result.
        box_short_edge_thresh (float): Minimum short side length to accept box.
        line_merge_y_thresh (float): Threshold to merge text boxes into lines by vertical center.

    Returns:
        Tuple of:
            - List[str]: merged OCR texts
            - List[Tuple[int, int, int, int]]: corresponding merged bounding boxes
    """
    ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    result = ocr_engine.ocr(np.array(image_pil), cls=True)

    filtered_results = []

    # Step 1: Filter boxes
    if result is None:
        return [], []
    for line in result:
        if line is None:
            continue
        for res in line:
            box = res[0]
            text = res[1][0]
            conf = res[1][1]

            edge_lengths = [
                np.linalg.norm(np.array(box[i]) - np.array(box[(i + 1) % 4]))
                for i in range(4)
            ]
            min_edge = min(edge_lengths)

            if conf >= confidence_threshold and min_edge >= box_short_edge_thresh:
                y_center = np.mean([pt[1] for pt in box])
                filtered_results.append({
                    "text": text,
                    "box": box,
                    "y_center": y_center
                })

    # Step 2: Group lines by vertical center
    filtered_results.sort(key=lambda x: x["y_center"])

    merged_lines = []
    current_line = []
    current_y = None

    for item in filtered_results:
        if current_y is None or abs(item["y_center"] - current_y) <= line_merge_y_thresh:
            current_line.append(item)
            current_y = item["y_center"] if current_y is None else (current_y + item["y_center"]) / 2
        else:
            merged_lines.append(current_line)
            current_line = [item]
            current_y = item["y_center"]
    if current_line:
        merged_lines.append(current_line)

    # Step 3: Convert each line into text + (x1, y1, x2, y2)
    final_texts = []
    final_boxes = []

    for line in merged_lines:
        merged_text = " ".join([x["text"] for x in line])
        all_points = np.concatenate([x["box"] for x in line], axis=0)
        x1 = int(np.min(all_points[:, 0]))
        y1 = int(np.min(all_points[:, 1]))
        x2 = int(np.max(all_points[:, 0]))
        y2 = int(np.max(all_points[:, 1]))
        final_texts.append(merged_text)
        final_boxes.append((x1, y1, x2, y2))

    return final_texts, final_boxes



def merge_sam_masks_with_ocr(sam_masks, ocr_boxes, image_size, iou_thresh=0.8, min_side=10):
    """
    Merge SAM masks based on OCR text regions using IoU.

    Args:
        sam_masks (list of dict): List of SAM masks, each with 'segmentation' (np.ndarray).
        ocr_boxes (list of tuple): List of (x1, y1, x2, y2) boxes.
        image_size (tuple): (width, height) of the image.
        iou_thresh (float): IoU threshold to consider a SAM mask overlapping with OCR region.
        min_side (int): Minimum size (both width and height) for OCR box to be kept.

    Returns:
        List of filtered/merged SAM masks.
    """

    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0.0
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = min(box1_area, box2_area)
        return inter_area / union_area

    final_masks = []
    for mask in sam_masks:
        seg = mask['segmentation']
        y_indices, x_indices = np.where(seg)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue
        box_mask = [min(x_indices), min(y_indices), max(x_indices), max(y_indices)]

        overlaps_ocr = any(compute_iou(box_mask, ocr_box) > iou_thresh for ocr_box in ocr_boxes)
        if not overlaps_ocr:
            final_masks.append(mask)

    for ocr_box in ocr_boxes:
        x1, y1, x2, y2 = map(int, ocr_box)
        width = x2 - x1
        height = y2 - y1

        # Skip too small boxes
        if width < min_side or height < min_side:
            continue

        mask_array = np.zeros((image_size[1], image_size[0]), dtype=bool)
        mask_array[y1:y2+1, x1:x2+1] = True
        area = (y2 - y1 + 1) * (x2 - x1 + 1)

        final_masks.append({
            'segmentation': mask_array,
            'area': area,
            'source': 'ocr'
        })

    return final_masks


def save_anns(image, anns, image_save_folder, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    # Convert RGBA float32 [0,1] to uint8 [0,255]
    img_uint8 = (img * 255).astype(np.uint8)
    overlay = Image.fromarray(img_uint8, mode='RGBA')
    base = image.convert('RGBA')
    result = Image.alpha_composite(base, overlay)

    result.save(f"{image_save_folder}/sam2_result.png")



def get_all_images(folder_path):
    """
    Walk through a folder and its subfolders to get all image file paths.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of absolute paths to all image files.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    image_paths = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.abspath(os.path.join(root, file)))
    
    return image_paths

def summarize_single_file(src_dir, filename):

    file_path = os.path.join(src_dir, filename, "anno.json")
    image_path = os.path.join(src_dir, filename, "sam2_result.png")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        key = data["image_uid"].split("__")[-1].split(".")[0]
        region_count = data["region_count"]
        org_img_path = data["image_path"]
        sam_path = image_path

        return key, {
            "region_count": region_count,
            "image_path": org_img_path,
            "sam_path": sam_path
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def mp_summarize(folder_path, save_name):

    all_files = os.listdir(folder_path)
    all_files = [f for f in all_files if os.path.isdir(os.path.join(folder_path, f))]

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(summarize_single_file, [(folder_path, f) for f in all_files])

    summary_json = {k: v for r in results if r for k, v in [r]}

    save_path = os.path.join(folder_path.strip("step2"), save_name + ".json")
    with open(save_path, "w") as f:
        json.dump(summary_json, f, indent=4)
    
    print(f"Readability Statistics saved to {save_path}. Stage 2 completed.")

def run_task(task_list, task_id, output_folder, prefix_remove="", sam2_checkpoint=None):
    gpu_id = task_id % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    desc = f"task: {task_id}"

    assert sam2_checkpoint is not None, "Please provide a valid SAM2 checkpoint path."
    print("load sam2...")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        points_per_batch=256,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.95,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.6,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=0.0,
        use_m2m=True,
    )

    for task_i in tqdm(task_list, desc):
        image_path = task_i["image_path"]
        image_uid = image_path.replace(prefix_remove, "").replace("/", "__")
        image_suffix = image_path.split(".")[-1]
        
        image_save_folder = image_uid.replace(f".{image_suffix}", "")
        
        image_save_folder = os.path.join(output_folder, image_save_folder)
        ann_json_path = f"{image_save_folder}/anno.json"
        if os.path.exists(ann_json_path):
            continue
        os.makedirs(image_save_folder, exist_ok=True)
        try:
            image_pil = Image.open(image_path).convert("RGB")
            image_width, image_height = image_pil.size
        except (OSError, ValueError) as e:
            print(f"Error opening image {image_path}: {e}")
            continue
        
        data_i = {
            "image_uid": image_uid,
            "image_path": image_path,
            "region_count": -1,
            "width": image_width,
            "height": image_height
        }
        try:
            masks = mask_generator.generate(np.array(image_pil))
        except Exception as e:
            print(f"Error generating masks for {image_path}: {e}")
            continue

        ocr_texts, ocr_boxes = call_ppocr(image_pil, image_width, image_height)
        # print(f" {image_path} before: {len(masks)} {ocr_texts}, len(ocr_boxes): {len(ocr_boxes)}")
        masks = merge_sam_masks_with_ocr(masks, ocr_boxes, (image_width, image_height))
        # print(f"after {len(masks)}")
        data_i["region_count"] = len(masks)

        save_anns(image_pil, masks, image_save_folder)
        
        # save json
        with open(ann_json_path, "w") as f:
            json.dump(data_i, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Remove background from images in a folder.")
    parser.add_argument("--sam2_ckpt", "-s", type=str, required=True, help="Path to the SAM2 checkpoint.")
    parser.add_argument("--input_folder", "-i", type=str, required=True, help="The path to the input folder.")
    parser.add_argument("--output_folder", "-o", type=str, required=True, help="The path to the output folder.")
    parser.add_argument("--save_name", type=str, default="step2_summarize", help="Name of the output summary file.")
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    sam2_ckpt = args.sam2_ckpt

    os.makedirs(output_folder, exist_ok=True)
    prefix_remove = input_folder if input_folder[-1] == "/" else input_folder + "/"
    
    task_list = get_all_images(input_folder)

    task_list = [{"image_path": image_path} for image_path in task_list]

    # download ppocr model in advance, cache it in advance
    _ = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

    # shuffle, to get rid of truncation or stopping...
    random_indices = np.random.permutation(len(task_list))
    task_list = [task_list[i] for i in random_indices]
    print(f"Total {len(task_list)} images.")
    
    num_gpus = torch.cuda.device_count()
    TASK_NUM = num_gpus if num_gpus > 0 else 1 

    per_task_num = len(task_list) // TASK_NUM
    
    t_list = []
    for t_idx in range(TASK_NUM):
        if t_idx == TASK_NUM - 1:
            if t_idx != 0:
                task_list_i = task_list[t_idx * per_task_num:]
            else:
                task_list_i = task_list
        else:
            task_list_i = task_list[t_idx * per_task_num: (t_idx + 1) * per_task_num]
        
        t_i = mp.Process(target=run_task, args=(task_list_i, t_idx, output_folder, prefix_remove, sam2_ckpt))
        t_i.start()
        t_list.append(t_i)
    
    for t_i in t_list:
        t_i.join()
    
    #mp_summarize(args.output_folder, args.save_name)

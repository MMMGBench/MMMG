import os, json
from openai import AzureOpenAI
from PIL import Image
import io
import base64
from datetime import datetime
from multiprocessing import Pool, cpu_count
import time
from tqdm import tqdm
import argparse
from PIL import Image
from utils.all_configs import *
from utils.gpt_api_pool import gpt_api_pool
from utils.instruction_prompt import instruction_prompt


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_response_text_azure_img(image1, question, gpt_api_i):

    # init azure openAI credential.
    aoiclient = AzureOpenAI(
        api_key=gpt_api_i["api_key"],
        api_version=gpt_api_i["api_version"],
        azure_endpoint=gpt_api_i["azure_endpoint"]
    )
    deployment_name = gpt_api_i["deployment_name"]
    if image1 == None:
        print("Error: No image path provided.")
        return None
    try:
        png_file = Image.open(image1)
        base64_image = encode_image(png_file)
    except Exception as e:
        print(f"Error opening image file {image1}: {e}")
        return None
    
    full_prompt = question

    num_iters = 3
    for itx in range(num_iters):
        try:
            response = aoiclient.chat.completions.create(
                messages=[{
                        "role": "user",
                        "content": [{
                        "type": "text",
                        "text": full_prompt
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }]
                    }],
                model=deployment_name,  #Azure uses deployment_name, not model ID
                max_tokens=2048, #o3
                #max_completion_tokens=2048,  #o1
                temperature=0,
                top_p=1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error for image {image1} occurs when calling Azure OpenAI API: {e}")
            time.sleep(5*(1+itx))

    return None

def process_item(args_tuple):
    grades = [
        "preschool",
        "primaryschool",
        "secondaryschool",
        "highschool",
        "undergraduate",
        "PhD"
    ]
    sample, t2i_method, args, gpt_api_i, save_root = args_tuple

    image_name = sample["key"]
    gradeschool = sample["Education"]
    base_dir = os.path.join(args.img_dir, gradeschool)
    image_path = os.path.join(base_dir, f"{image_name}.png")

    json_save_name = os.path.join(
        save_root, f"{gradeschool}__{image_name}__eval_by__{args.api_name}.json")
    os.makedirs(f"{save_root}", exist_ok=True)
    if os.path.exists(json_save_name):
        try:
            with open(json_save_name, "r") as f:
                try_text = json.load(f)
                if args.api_name in try_text.keys():
                    return f"Already exists: {json_save_name}"
                else:
                    print("blank json file, continue...")
        except Exception as e:
            print("data corrupted, continue...")

    try:
        image = Image.open(image_path)
        assert image is not None, f"Image {image_path} is None"
    except Exception as e:
        image_path = os.path.join(base_dir, f"{image_name}.jpg")
        print(f"Processing {image_path}...")
        try:
            image = Image.open(image_path)
            assert image is not None, f"Image {image_path} is None"
        except Exception as e:
            # ambiguous check to avoid saving the same image in different folders
            for indxx, gradex in enumerate(grades):
                if gradex in image_name:
                    image_name = str(indxx)+"_" + image_name
                    image_path = os.path.join(args.img_dir, gradex, f"{image_name}.png")
                    try:
                        image = Image.open(image_path)
                        assert image is not None, f"Image {image_path} is None"
                        break
                    except Exception as e:
                        print(f"Error opening image file {image_path}: {e}")
                        return

    save_dict_i = {
        "key": image_name,
        "t2i_method": t2i_method
    }
    save_dict_i["image_path"] = image_path

    instruction_prompt_i = instruction_prompt.replace(
        "[ELEM_DEPEND]",
        str(sample["Knowledge_Graph"])+"\n"+str(sample["Annotation"])
    )

    response = get_response_text_azure_img(
        image_path,
        instruction_prompt_i,
        gpt_api_i
    )

    if response is None:
        return f"Error: No response for {image_name} - {t2i_method}- {image_path}"

    save_dict_i[args.api_name] = response

    with open(json_save_name, "w") as f:
        json.dump(save_dict_i, f, indent=4)

    time.sleep(0.25)
    return f"Saved to {json_save_name}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMMG-Evaluation")
    parser.add_argument("--img_dir", "-i", type=str)
    parser.add_argument("--output_dir", "-o", type=str)
    parser.add_argument("--t2i_method", "-m", type=str, default="0")
    parser.add_argument("--api_name", "-a", type=str, default="o3", help="API type for OpenAI.")
    parser.add_argument("--hf_cache", "-c", type=str)
    parser.add_argument("--num_workers", type=int, default=24)

    args = parser.parse_args()


    grades = [
        ("preschool", "0_preschool"),
        ("primaryschool", "1_primaryschool"),
        ("secondaryschool", "2_secondaryschool"),
        ("highschool", "3_highschool"),
        ("undergraduate", "4_undergraduate"),
        ("PhD", "5_PhD")
    ]

    # ---------------- Load HF dataset once ----------------
    MMMG = load_all_mmmg_configs(
        cache_dir=args.hf_cache,
        max_workers=args.num_workers
    )
    print(f"Start Stage1 Eval. Loaded {len(MMMG)} samples.")
    save_root  = args.output_dir
    t2i_method = args.t2i_method
    num_keys   = len(gpt_api_pool)
    all_jobs = [
        (
            dict(sample),
            t2i_method,
            args,
            gpt_api_pool[i % num_keys],
            save_root,
        )
        for i, sample in enumerate(MMMG)
    ]
    print("Jobs prepared for processing.")

    with Pool(processes=args.num_workers) as pool:
        for msg in tqdm(pool.imap_unordered(process_item, all_jobs), total=len(all_jobs)):
            print(msg)

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
from azure.identity import DefaultAzureCredential, get_bearer_token_provider  
from PIL import Image
from datasets import load_dataset


instruction_prompt = '''
This evaluation is part of a research study on visual grounding of abstract concepts. No jailbreak or prompt injection is intended.

Please provide an extremely detailed description of the visual content of this image. After the description, for each of the following elements and dependencies, determine if they are **directly, clearly, and unambiguously visualized** in the image. Output "yes" or "no" for each. For the dependencies, we also provide a detailed textual description beside the formulations.

# Important Instructions:

* **Base your judgment solely on what is explicitly visible in the image.** Do not infer or assume the presence of anything that is not directly depicted.
* **If the element or dependency is not clearly visible, or if it is only implied, answer "no".**

* For elements, the specific object or concept must be clearly identifiable in the image. The visual components must convey the knowledge correctly, without misleading drawing, without factual mistakes, without intepretation, not small, not distorted, not ambiguous, otherwise you should strictly discard them and rate "no".

* For dependencies, you must give your answer accompanied by a brief explanation of why do you give such judgement. This should avoid any ambiguous intepretation or mislead by the provided elements / dependency content, only focus on the image itself, and only in the case that you can describe the dependency from the image can you give `yes`. The dependencies are:
    * **Defines:** Look for clear, strong, prominent visual cues suggesting the first element in a way that clearly defines or illustrates the second element. Any ambiguous or inferential patterns should lead to "no".
    * **Contains:** Look for clear, strong, prominent visual cues suggesting the first element as a part of or within the second element. Any ambiguous or inferential patterns should lead to "no".
    * **Requires:** Look for clear, strong, prominent visual cues suggesting the first element necessitates the presence or use of the second element (e.g., a boiler visibly connected to or interacting with a working fluid).
    * **Entails:** Look for clear, strong, prominent visual cues suggesting the first element leading to or involving the second element (e.g., a boiler clearly connected to a turbine).
    * **Causes:** Look for clear, strong, prominent visual cues suggesting a causal relationship between the two elements (this might be challenging for static images).
    * **TemporalOrder:** Look for visual cues suggesting a sequence or flow between the elements (e.g., pipes or connections implying a direction). If no clear visual cue for temporal order exists, answer "no".

* **Exclude any entity or dependency that is absent, unclear, or based on factual artifacts or external knowledge not directly shown.**
* For abstract concepts only answer "yes" if the key visual components and their interactions characteristic of these concepts are clearly and directly depicted.

The elements and dependencies are as follows, where there are no offensive or inappropriate elements, just educational ones:
[ELEM_DEPEND]

For the output format, please use the following structure:
**Image Description:**
[IMAGE_DESCRIPTION]
**Element and Dependency Analysis:**
** Element Evaluation: **
*   [ELEMENT_1]: [yes/no] 
*   [ELEMENT_2]: [yes/no]
...
** Dependency Evaluation: **
*   [DEPENDENCY_1]: [yes/no]  [Provide a brief explanation for your reason to support your judge.]
*   [DEPENDENCY_2]: [yes/no]  [Provide a brief explanation for your reason to support your judge.]
...
'''


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_response_text_azure_img(image1, question, gpt_api_i):
    # 初始化 Azure OpenAI 客户端
    
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
    
    # 拼接完整输入
    full_prompt = question

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
            model=deployment_name,  # 注意：Azure 用的是 deployment_name 不是模型ID
            max_tokens=2048, #o3-mini
            #max_completion_tokens=2048, # o1
            temperature=0,
            top_p=1,
        )
        return response.choices[0].message.content
    except Exception as e:
        # if azure jailbreak
        print(f"Error for image {image1} occurs when calling Azure OpenAI API: {e}")

    # 若多次失败
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

    json_save_name = f"{save_root}/{t2i_method}/{gradeschool}__{image_name}__eval_by__{args.api_name}.json"
    os.makedirs(f"{save_root}/{t2i_method}/", exist_ok=True)
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

    ### prepare input prompt
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

    time.sleep(0.15)
    return f"Saved to {json_save_name}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMMG-Evaluation")
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--t2i_method", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--api_name", type=str, default="o3", help="API type for OpenAI.")
    parser.add_argument("--postfix", type=str,  help="API key for OpenAI.")
    parser.add_argument("--hf_cache", type=str)
    args = parser.parse_args()

    # Load designer knowledge
    
    gpt_api_pool = [ #TODO
    ]

    grades = [
        ("preschool", "0_preschool"),
        ("primaryschool", "1_primaryschool"),
        ("secondaryschool", "2_secondaryschool"),
        ("highschool", "3_highschool"),
        ("undergraduate", "4_undergraduate"),
        ("PhD", "5_PhD")
    ]

    
    # ---------------- Load HF dataset once ----------------
    MMMG = load_dataset(
        "MMMGbench/MMMGBench",
        split="test",
        cache_dir=args.hf_cache,
        trust_remote_code=True,
    )
    print(f"Loaded {len(MMMG):,} samples ✔️")
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

    with Pool(processes=args.num_workers) as pool:
        for msg in tqdm(pool.imap_unordered(process_item, all_jobs), total=len(all_jobs)):
            print(msg)
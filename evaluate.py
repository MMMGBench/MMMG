import argparse
import subprocess
import os
from mmmg_eval.utils.gpt_api_pool import gpt_api_pool

def run_cmd(cmd):
    print(f"[Running] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def main():
    parser = argparse.ArgumentParser(description="Unified MMMG Evaluation Pipeline")
    parser.add_argument("-i", "--img_dir", required=True, help="Directory containing generated images")
    parser.add_argument("-o", "--output_dir", default = "./output/", help="Directory to save results")
    parser.add_argument("-s", "--sam2_ckpt", required=True, help="Path to SAM2.1 checkpoint")
    parser.add_argument("-m", "--t2i_method", required=True, help="Name of the T2I method")
    parser.add_argument("-a", "--api_name", required=True, help="Name of the OpenAI evaluator API method")
    parser.add_argument("-c", "--hf_cache", default="./data/MMMG", help="HuggingFace cache path (optional)")

    args = parser.parse_args()
    api_name = args.api_name
    t2i_model = args.t2i_method
    img_dir = args.img_dir
    out_dir = os.path.join(args.output_dir, t2i_model)
    sam2_ckpt = args.sam2_ckpt   
    hf_cache = args.hf_cache
    num_workers = min(40, len(gpt_api_pool)*10)

    os.makedirs(out_dir, exist_ok=True)

    # Step 1 – Knowledge Fidelity
    cmd1 = f"python mmmg_eval/step1_knowledge_integrity.py -i {img_dir} -o {out_dir}/step1 -m {t2i_model} -a {api_name} -c {hf_cache} --num_workers {num_workers}"
    run_cmd(cmd1)

    # Formulate Knowledge Fidelity result into a JSON file
    cmd2 = f"python mmmg_eval/utils/stat_knowledge.py --result_folder {out_dir}/step1 --image_folder {img_dir} --api_name {api_name} --output_dir {out_dir} --save_name {t2i_model}_step1_summarize"
    run_cmd(cmd2)

    # Step 2 – Visual Readability
    cmd3 = f"python mmmg_eval/step2_readability.py -s {sam2_ckpt} -i {img_dir} -o {out_dir}/step2 --save_name {t2i_model}_step2_summarize"
    run_cmd(cmd3)

    # Formulate Readability result into a JSON file
    cmd4 = f"python mmmg_eval/utils/stat_sam2.py -s {out_dir}/step2 -o {out_dir} -n {t2i_model}_step2_summarize"
    run_cmd(cmd4)

    # Step 3 – Final Score Aggregation
    cmd5 = f"python mmmg_eval/step3_stat.py --data_dir {out_dir}/{t2i_model}_step1_summarize.json --score_dir {out_dir}/{t2i_model}_step2_summarize.json --save_dir {out_dir}/{t2i_model}_MMMGStat.json"
    run_cmd(cmd5)

    print(f"\n✅ Evaluation complete. Final results saved in {out_dir}/final")

if __name__ == "__main__":
    main()

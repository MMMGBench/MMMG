# MMMG-Eval

Evaluation code for the **M**assive **M**ulti-discipline **M**ulti-tier Knowledge-Image **G**eneration benchmark.

* **Dataset:** https://huggingface.co/datasets/MMMGBench/MMMG  
* **Metric:** MMMG-Score = Knowledge Fidelity (1 − GED) × Visual Readability (SAM 2.1).

---

## 1  Install

```bash
git clone https://github.com/MMMGBench/MMMG.git
cd MMMG
conda env create -f environment.yaml
```

## 2 Prepare data
```bash
/data/
 ├─ preschool/
 ├─ primaryschool/
 ├─ secondaryschool/
 ├─ highschool/
 ├─ undergraduate/
 └─ PhD/
 ```
Each folder holds your model’s generated images (<prompt_key>.png).

## 3 Run evaluation
We use AzureOpenAI service. If you adopt your API code from OpenAI websete, please **Modify mmmg_eval/step1_knowledge_integrity.py**.

Please fill in your API keys into mmmg_eval/utils/gpt_api_pool.py.

Then run the following script:
```bash
python evaluate.py \
--img_dir IMG_FOLDER \
--output_dir BASE_OUTPUT_FOLDER \
--sam2_ckpt SAM2_CKPT_PATH \
--t2i_method T2I_MODEL_NAME \
--api_name OpenAI_MODEL_NAME \
--hf_cache HF_CACHE

```

For example, benchmarking GPT-4o Image generation:
```bash
python evaluate.py \
--img_dir ./data/GPT-4o \
--output_dir ./output \
--sam2_ckpt /YOUR/PATH/TO/sam2/checkpoints/sam2.1_hiera_large.pt \
--t2i_method GPT-4o \
--api_name o3 \
--hf_cache ./data/MMMG
```

## 📖 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{luo2025mmmg,
  title={MMMG: A Massive, Multidisciplinary, Multi-Tier Generation Benchmark for Text-to-Image Reasoning},
  author={Yuxuan Luo and Yuhui Yuan and Junwen Chen and Haonan Cai and Ziyi Yue and Yuwei Yang and Fatima Zohra Daha and Ji Li and Zhouhui Lian},
  journal={arXiv preprint arXiv:2506.10963},
  year={2025}
}

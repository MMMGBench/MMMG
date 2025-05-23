# MMMG-Eval

Evaluation code for the **M**assive **M**ulti-discipline **M**ulti-tier Knowledge-Image **G**eneration benchmark.

* **Dataset:** https://huggingface.co/datasets/MMMGBench/MMMGBench  
* **Metric:** MMMG-Score = Knowledge Fidelity (1 − GED) × Visual Readability (SAM 2.1).

---

## 1  Install

```bash
conda create -n mmmg python=3.10 -y
conda activate mmmg
pip install -r requirements.txt
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
```bash
IMG_DIR=/data
OUT_DIR=./results
SAM2=/path/to/sam2.1_hiera_large.pt
MODEL=MyT2I
HF_CACHE=~/.cache/mmmg

# step 1 – knowledge fidelity
python mmmg-eval/step1_knowledge_integrity.py \
       --img_dir $IMG_DIR --output_dir $OUT_DIR/step1 \
       --t2i_method $MODEL  --hf_cache $HF_CACHE

# step 2 – visual readability
python mmmg-eval/step2_readability.py \
       --sam2_ckpt $SAM2 -i $IMG_DIR -o $OUT_DIR/step2

# step 3 – final score
python mmmg-eval/step3_stat.py \
       --dir1 $OUT_DIR/step1 --dir2 $OUT_DIR/step2 \
       --output_dir $OUT_DIR/final
```
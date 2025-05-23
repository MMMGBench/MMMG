from datasets import load_dataset
import json
cache_root = "/detr_blob/v-luoyuxuan/hf_cache"
ds = load_dataset("MMMGbench/MMMGBench", split="test", cache_dir=cache_root, trust_remote_code=True)

all_jobs = [
        (
            dict(sample),   
            
        )
        for i, sample in enumerate(ds)
    ]

print(all_jobs[0])
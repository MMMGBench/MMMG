from datasets import load_dataset, concatenate_datasets
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_all_mmmg_configs(cache_dir="~/.cache/mmmg", max_workers=8):
    """
    Load and concatenate all config splits from the MMMGbench/MMMG dataset.

    Args:
        cache_dir (str): HuggingFace cache directory.
        max_workers (int): Maximum number of parallel threads to use.

    Returns:
        Dataset: The concatenated dataset with all config entries.
    """
    all_configs = [
        'PhD_Biology', 'PhD_Chemistry', 'PhD_Economics', 'PhD_Engineering', 'PhD_Geography',
        'PhD_History', 'PhD_Literature', 'PhD_Math', 'PhD_Philosophy', 'PhD_Sociology',
        'highschool_Biology', 'highschool_Chemistry', 'highschool_Economics', 'highschool_Engineering',
        'highschool_Geography', 'highschool_History', 'highschool_Literature', 'highschool_Math',
        'highschool_Philosophy', 'highschool_Sociology', 'preschool_Biology', 'preschool_Chemistry',
        'preschool_Economics', 'preschool_Engineering', 'preschool_Geography', 'preschool_History',
        'preschool_Literature', 'preschool_Math', 'preschool_Sociology', 'primaryschool_Biology',
        'primaryschool_Chemistry', 'primaryschool_Economics', 'primaryschool_Engineering',
        'primaryschool_Geography', 'primaryschool_History', 'primaryschool_Literature',
        'primaryschool_Math', 'primaryschool_Philosophy', 'primaryschool_Sociology',
        'secondaryschool_Biology', 'secondaryschool_Chemistry', 'secondaryschool_Economics',
        'secondaryschool_Engineering', 'secondaryschool_Geography', 'secondaryschool_History',
        'secondaryschool_Literature', 'secondaryschool_Math', 'secondaryschool_Philosophy',
        'secondaryschool_Sociology', 'undergraduate_Biology', 'undergraduate_Chemistry',
        'undergraduate_Economics', 'undergraduate_Engineering', 'undergraduate_Geography',
        'undergraduate_History', 'undergraduate_Literature', 'undergraduate_Math',
        'undergraduate_Philosophy', 'undergraduate_Sociology'
    ]

    expanded_cache_dir = os.path.expanduser(cache_dir)

    def load_one_config(cfg_name):
        print(f"Loading config: {cfg_name}...")
        return load_dataset(
            "MMMGbench/MMMG",
            name=cfg_name,
            split="test",
            cache_dir=expanded_cache_dir,
            trust_remote_code=True,
        )

    all_datasets = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_one_config, cfg): cfg for cfg in all_configs}
        for future in as_completed(futures):
            cfg = futures[future]
            try:
                ds = future.result()
                all_datasets.append(ds)
            except Exception as e:
                print(f"❌ Failed to load {cfg}: {e}")

    full_dataset = concatenate_datasets(all_datasets)
    print(f"✅ Loaded total {len(full_dataset):,} samples from {len(all_datasets)} configs.")
    return full_dataset

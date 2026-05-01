from huggingface_hub import HfFileSystem

"""
Default configuration constants for the kernel-operator playground.
"""

DATASETS = {
    "eval":     "Salesforce/GiftEvalParquet",
    "pretrain": "Salesforce/GiftEvalPretrain",
}

# 1. Fetch the Pretrain subset names dynamically
fs = HfFileSystem()
root_items = fs.ls(f"datasets/{DATASETS['pretrain']}", detail=True)
pretrain_subsets = [
    item['name'].split('/')[-1] 
    for item in root_items 
    if item['type'] == 'directory' and not item['name'].endswith('.gitattributes')
]

# 2. Create a mapping of { "Name": ("Repo", "Subset") }
# This tells your downstream code exactly which dataset to pull from
CONFIG_MAP = {
    # Eval Parquet Configs
    "Energy":  (DATASETS["eval"], "electricity_H_long"),
    "Cloud":   (DATASETS["eval"], "bitbrains_fast_storage_5T_long"),
    "Traffic": (DATASETS["eval"], "loop_seattle_H_long"),
    "Solar":   (DATASETS["eval"], "solar_H_long"),
}

# Add all Pretrain configs to the map automatically
for subset in pretrain_subsets:
    CONFIG_MAP[f"Pretrain_{subset}"] = (DATASETS["pretrain"], subset)

DEFAULT_CONFIG = {
    # -- dataset -----------------------------------------------
    # Now, simply specify which key you want to load from the map
    "ACTIVE_CONFIG_KEY": "Energy", # Change to e.g., "Pretrain_covid_mobility" to switch
    "DATASET_NAME": DATASETS["eval"],
    "DATASETS": DATASETS,
    "CONFIG_MAP": CONFIG_MAP,

    # -- sampling ----------------------------------------------
    "N_SAMPLES_TO_LOAD": 5000,
    "N_TEST_SAMPLES": 200,
    "RANDOM_SEED": 0,

    # -- kernel operator ---------------------------------------
    "GAMMA": 1e-2,
    "PAIRWISE_MEDIAN_SUBSET": 24,

    # -- plotting ----------------------------------------------
    "HISTORY_CONTEXT_TO_PLOT": 240,
    "SAVE_FIGURE": True,
    "SHOW_FIGURES": True,
    "OUTPUT_STEM": "kernel_operator_forecast",
}
from datasets import get_dataset_config_names

"""
Default configuration constants for the kernel-operator playground.

Import and override as needed:
    from utils.config import DEFAULT_CONFIG
    cfg = {**DEFAULT_CONFIG, "N_SAMPLES_TO_LOAD": 2000}
"""

DATASETS = {
    "eval":     "Salesforce/GiftEvalParquet",
    "pretrain": "Salesforce/GiftEvalPretrain",
}

# Dynamically fetch all 153 subset names from the GiftEvalPretrain repo
# (e.g., 'borg_cluster_data_2011', 'buildings_900k', 'kaggle_web_traffic_weekly')
pretrain_subsets = get_dataset_config_names(DATASETS["pretrain"])

# Create a dictionary of { "Subset_Name": "Subset_Name" } for the pretrain data
PRETRAIN_CONFIGS = {name: name for name in pretrain_subsets}

DEFAULT_CONFIG = {
    # -- dataset -----------------------------------------------
    # Swap this depending on which dataset you want to load by default
    "DATASET_NAME": DATASETS["pretrain"], 
    "DATASETS": DATASETS,
    
    # Merge your original eval configs with all 153 pretrain configs
    "CONFIGS": {
        "Energy": "electricity_H_long",
        "Cloud":  "bitbrains_fast_storage_5T_long",
        "Traffic": "loop_seattle_H_long",
        "Solar":  "solar_H_long",
        **PRETRAIN_CONFIGS 
    },

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
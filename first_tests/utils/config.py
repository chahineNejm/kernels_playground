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

DEFAULT_CONFIG = {
    # -- dataset -----------------------------------------------
    "DATASET_NAME": DATASETS["eval"],
    "DATASETS": DATASETS,
    "CONFIGS": {
        "Energy": "electricity_H_long",
        "Cloud":   "bitbrains_fast_storage_5T_long",
        "Traffic": "loop_seattle_H_long",
        "Solar":   "solar_H_long",
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

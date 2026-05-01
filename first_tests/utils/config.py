"""
Default configuration constants for the kernel-operator playground.

Two datasets are available:
    eval     → Salesforce/GiftEvalParquet   (97 configs, for evaluation)
    pretrain → Salesforce/GiftEvalPretrain  (single "default" config, for training)

Switch datasets:
    from utils.config import DEFAULT_CONFIG, DATASETS
    DEFAULT_CONFIG["DATASET_NAME"] = DATASETS["pretrain"]

Or pass dataset_name= to any function:
    build_examples(config="electricity_H_long", dataset_name=DATASETS["pretrain"])
"""

# ===================================================================
# DATASET REPOS
# ===================================================================

DATASETS = {
    "eval":     "Salesforce/GiftEvalParquet",
    "pretrain": "Salesforce/GiftEvalPretrain",
}

# ===================================================================
# EVAL CONFIGS  (GiftEvalParquet — 97 configs)
# ===================================================================

EVAL_CONFIGS = {
    # --- Energy ---
    "Energy":           "electricity_H_long",
    "Solar":            "solar_H_long",
    # --- Web / CloudOps ---
    "Cloud":            "bitbrains_fast_storage_5T_long",
    # --- Transport ---
    "Traffic":          "loop_seattle_H_long",
}

# Full list of all 97 eval configs (for reference / iteration)
ALL_EVAL_CONFIGS = [
    # bitbrains
    "bitbrains_fast_storage_5T_long", "bitbrains_fast_storage_5T_medium",
    "bitbrains_fast_storage_5T_short", "bitbrains_fast_storage_H_short",
    "bitbrains_rnd_5T_long", "bitbrains_rnd_5T_medium",
    "bitbrains_rnd_5T_short", "bitbrains_rnd_H_short",
    # bizitobs
    "bizitobs_application_10S_long", "bizitobs_application_10S_medium",
    "bizitobs_application_10S_short",
    "bizitobs_l2c_5T_long", "bizitobs_l2c_5T_medium", "bizitobs_l2c_5T_short",
    "bizitobs_l2c_H_long", "bizitobs_l2c_H_medium", "bizitobs_l2c_H_short",
    "bizitobs_service_10S_long", "bizitobs_service_10S_medium",
    "bizitobs_service_10S_short",
    # car parts
    "car_parts_M_short",
    # covid
    "covid_deaths_D_short",
    # electricity
    "electricity_15T_long", "electricity_15T_medium", "electricity_15T_short",
    "electricity_D_short",
    "electricity_H_long", "electricity_H_medium", "electricity_H_short",
    "electricity_W_short",
    # ett1
    "ett1_15T_long", "ett1_15T_medium", "ett1_15T_short",
    "ett1_D_short",
    "ett1_H_long", "ett1_H_medium", "ett1_H_short",
    "ett1_W_short",
    # ett2
    "ett2_15T_long", "ett2_15T_medium", "ett2_15T_short",
    "ett2_D_short",
    "ett2_H_long", "ett2_H_medium", "ett2_H_short",
    "ett2_W_short",
    # hierarchical sales
    "hierarchical_sales_D_short", "hierarchical_sales_W_short",
    # hospital
    "hospital_M_short",
    # jena weather
    "jena_weather_10T_long", "jena_weather_10T_medium", "jena_weather_10T_short",
    "jena_weather_D_short",
    "jena_weather_H_long", "jena_weather_H_medium", "jena_weather_H_short",
    # kdd cup
    "kdd_cup_2018_D_short",
    "kdd_cup_2018_H_long", "kdd_cup_2018_H_medium", "kdd_cup_2018_H_short",
    # loop seattle (traffic)
    "loop_seattle_5T_long", "loop_seattle_5T_medium", "loop_seattle_5T_short",
    "loop_seattle_D_short",
    "loop_seattle_H_long", "loop_seattle_H_medium", "loop_seattle_H_short",
    # m4
    "m4_daily_D_short", "m4_hourly_H_short", "m4_monthly_M_short",
    "m4_quarterly_Q_short", "m4_weekly_W_short", "m4_yearly_A_short",
    # m_dense
    "m_dense_D_short",
    "m_dense_H_long", "m_dense_H_medium", "m_dense_H_short",
    # restaurant
    "restaurant_D_short",
    # saugeen
    "saugeen_D_short", "saugeen_M_short", "saugeen_W_short",
    # solar
    "solar_10T_long", "solar_10T_medium", "solar_10T_short",
    "solar_D_short",
    "solar_H_long", "solar_H_medium", "solar_H_short",
    "solar_W_short",
    # sz taxi
    "sz_taxi_15T_long", "sz_taxi_15T_medium", "sz_taxi_15T_short",
    "sz_taxi_H_short",
    # temperature / rain
    "temperature_rain_D_short",
    # us births
    "us_births_D_short", "us_births_M_short", "us_births_W_short",
]

# ===================================================================
# PRETRAIN CONFIGS  (GiftEvalPretrain — single "default" config)
# ===================================================================
# The pretrain dataset has one config ("default") containing all series.
# Pass dataset_name=DATASETS["pretrain"] and config="default".

PRETRAIN_CONFIG = "default"

# ===================================================================
# DEFAULT CONFIG
# ===================================================================

DEFAULT_CONFIG = {
    # -- dataset -----------------------------------------------
    "DATASET_NAME": DATASETS["eval"],
    "DATASETS": DATASETS,
    "CONFIGS": EVAL_CONFIGS,                # friendly-name → HF config
    "ALL_EVAL_CONFIGS": ALL_EVAL_CONFIGS,   # full flat list
    "PRETRAIN_CONFIG": PRETRAIN_CONFIG,

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

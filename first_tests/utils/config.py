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
# PRETRAIN SUBSETS  (GiftEvalPretrain — each is a subdirectory)
# ===================================================================
# Each subset is loaded via data_files="{subset}/*.arrow".
# Pass dataset_name=DATASETS["pretrain"] and config="{subset_name}".

ALL_PRETRAIN_SUBSETS = [
    # transport
    "BEIJING_SUBWAY_30MIN", "HZMETRO", "LOS_LOOP", "PEMS03", "PEMS04",
    "PEMS07", "PEMS08", "PEMS_BAY", "Q-TRAFFIC", "SHMETRO",
    "pedestrian_counts", "rideshare_with_missing", "taxi_30min",
    "traffic_hourly", "traffic_weekly", "uber_tlc_daily", "uber_tlc_hourly",
    "vehicle_trips_with_missing",
    # cloud / web
    "alibaba_cluster_trace_2018", "azure_vm_traces_2017",
    "borg_cluster_data_2011", "godaddy",
    "extended_web_traffic_with_missing", "kaggle_web_traffic_weekly",
    "wiki-rolling_nips",
    # energy
    "australian_electricity_demand", "covid19_energy", "elecdemand", "elf",
    "gfc12_load", "gfc14_load", "gfc17_load",
    "largest_2017", "largest_2018", "largest_2019", "largest_2020", "largest_2021",
    "lcl", "london_smart_meters_with_missing",
    "residential_load_power", "residential_pv_power",
    "solar_power", "wind_farms_with_missing", "wind_power",
    # buildings
    "bdg-2_bear", "bdg-2_fox", "bdg-2_panther", "bdg-2_rat",
    "buildings_900k",
    # weather / nature
    "beijing_air_quality", "borealis", "china_air_quality",
    "oikolab_weather", "sceaux", "smart", "subseasonal", "subseasonal_precip",
    "sunspot_with_missing", "temperature_rain_D_short",
    # climate (CMIP6)
    "cmip6_1850", "cmip6_1855", "cmip6_1860", "cmip6_1865", "cmip6_1870",
    "cmip6_1875", "cmip6_1880", "cmip6_1885", "cmip6_1890", "cmip6_1895",
    "cmip6_1900", "cmip6_1905", "cmip6_1910", "cmip6_1915", "cmip6_1920",
    "cmip6_1925", "cmip6_1930", "cmip6_1935", "cmip6_1940", "cmip6_1945",
    "cmip6_1950", "cmip6_1955", "cmip6_1960", "cmip6_1965", "cmip6_1970",
    "cmip6_1975", "cmip6_1980", "cmip6_1985", "cmip6_1990", "cmip6_1995",
    "cmip6_2000", "cmip6_2005", "cmip6_2010",
    # climate (ERA5)
    "era5_1989", "era5_1990", "era5_1991", "era5_1992", "era5_1993",
    "era5_1994", "era5_1995", "era5_1996", "era5_1997", "era5_1998",
    "era5_1999", "era5_2000", "era5_2001", "era5_2002", "era5_2003",
    "era5_2004", "era5_2005", "era5_2006", "era5_2007", "era5_2008",
    "era5_2009", "era5_2010", "era5_2011", "era5_2012", "era5_2013",
    "era5_2014", "era5_2015", "era5_2016", "era5_2017", "era5_2018",
    # healthcare
    "cdc_fluview_ilinet", "cdc_fluview_who_nrevss", "covid_mobility",
    "project_tycho",
    # sales
    "cockatoo", "favorita_sales", "favorita_transactions", "m5",
    # economics / finance
    "bitcoin_with_missing", "bull", "fred_md", "hog", "ideal",
    # benchmarks (M1/M3)
    "m1_monthly", "m1_quarterly", "m1_yearly",
    "monash_m3_monthly", "monash_m3_other", "monash_m3_quarterly", "monash_m3_yearly",
    # other
    "cif_2016_12", "cif_2016_6", "kdd2022",
    "nn5_daily_with_missing", "nn5_weekly",
    "pdb", "spain",
    "tourism_monthly", "tourism_quarterly", "tourism_yearly",
]

# ===================================================================
# EVAL — grouped by domain
# ===================================================================

EVAL_BY_DOMAIN = {
    "Energy": [
        "electricity_15T_long", "electricity_15T_medium", "electricity_15T_short",
        "electricity_D_short",
        "electricity_H_long", "electricity_H_medium", "electricity_H_short",
        "electricity_W_short",
        "ett1_15T_long", "ett1_15T_medium", "ett1_15T_short",
        "ett1_D_short", "ett1_H_long", "ett1_H_medium", "ett1_H_short", "ett1_W_short",
        "ett2_15T_long", "ett2_15T_medium", "ett2_15T_short",
        "ett2_D_short", "ett2_H_long", "ett2_H_medium", "ett2_H_short", "ett2_W_short",
        "solar_10T_long", "solar_10T_medium", "solar_10T_short",
        "solar_D_short", "solar_H_long", "solar_H_medium", "solar_H_short", "solar_W_short",
    ],
    "Transport": [
        "loop_seattle_5T_long", "loop_seattle_5T_medium", "loop_seattle_5T_short",
        "loop_seattle_D_short",
        "loop_seattle_H_long", "loop_seattle_H_medium", "loop_seattle_H_short",
        "sz_taxi_15T_long", "sz_taxi_15T_medium", "sz_taxi_15T_short", "sz_taxi_H_short",
    ],
    "Web/CloudOps": [
        "bitbrains_fast_storage_5T_long", "bitbrains_fast_storage_5T_medium",
        "bitbrains_fast_storage_5T_short", "bitbrains_fast_storage_H_short",
        "bitbrains_rnd_5T_long", "bitbrains_rnd_5T_medium",
        "bitbrains_rnd_5T_short", "bitbrains_rnd_H_short",
        "bizitobs_application_10S_long", "bizitobs_application_10S_medium",
        "bizitobs_application_10S_short",
        "bizitobs_l2c_5T_long", "bizitobs_l2c_5T_medium", "bizitobs_l2c_5T_short",
        "bizitobs_l2c_H_long", "bizitobs_l2c_H_medium", "bizitobs_l2c_H_short",
        "bizitobs_service_10S_long", "bizitobs_service_10S_medium",
        "bizitobs_service_10S_short",
    ],
    "Nature": [
        "jena_weather_10T_long", "jena_weather_10T_medium", "jena_weather_10T_short",
        "jena_weather_D_short",
        "jena_weather_H_long", "jena_weather_H_medium", "jena_weather_H_short",
        "kdd_cup_2018_D_short",
        "kdd_cup_2018_H_long", "kdd_cup_2018_H_medium", "kdd_cup_2018_H_short",
        "saugeen_D_short", "saugeen_M_short", "saugeen_W_short",
        "temperature_rain_D_short",
        "us_births_D_short", "us_births_M_short", "us_births_W_short",
    ],
    "Sales": [
        "car_parts_M_short",
        "hierarchical_sales_D_short", "hierarchical_sales_W_short",
        "restaurant_D_short",
    ],
    "Healthcare": [
        "covid_deaths_D_short",
        "hospital_M_short",
    ],
    "Econ/Finance": [
        "m4_daily_D_short", "m4_hourly_H_short", "m4_monthly_M_short",
        "m4_quarterly_Q_short", "m4_weekly_W_short", "m4_yearly_A_short",
        "m_dense_D_short", "m_dense_H_long", "m_dense_H_medium", "m_dense_H_short",
    ],
}

# ===================================================================
# EVAL — grouped by frequency
# ===================================================================

EVAL_BY_FREQ = {
    "10S": [c for c in ALL_EVAL_CONFIGS if "_10S_" in c],
    "5T":  [c for c in ALL_EVAL_CONFIGS if "_5T_"  in c],
    "10T": [c for c in ALL_EVAL_CONFIGS if "_10T_" in c],
    "15T": [c for c in ALL_EVAL_CONFIGS if "_15T_" in c],
    "H":   [c for c in ALL_EVAL_CONFIGS if "_H_"   in c],
    "D":   [c for c in ALL_EVAL_CONFIGS if "_D_"   in c],
    "W":   [c for c in ALL_EVAL_CONFIGS if "_W_"   in c],
    "M":   [c for c in ALL_EVAL_CONFIGS if "_M_"   in c],
    "Q":   [c for c in ALL_EVAL_CONFIGS if "_Q_"   in c],
    "A":   [c for c in ALL_EVAL_CONFIGS if "_A_"   in c],
}

# ===================================================================
# PRETRAIN — grouped by domain
# ===================================================================

PRETRAIN_BY_DOMAIN = {
    "Transport": [
        "BEIJING_SUBWAY_30MIN", "HZMETRO", "LOS_LOOP",
        "PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMS_BAY",
        "Q-TRAFFIC", "SHMETRO",
        "pedestrian_counts", "rideshare_with_missing", "taxi_30min",
        "traffic_hourly", "traffic_weekly",
        "uber_tlc_daily", "uber_tlc_hourly",
        "vehicle_trips_with_missing",
    ],
    "Web/CloudOps": [
        "alibaba_cluster_trace_2018", "azure_vm_traces_2017",
        "borg_cluster_data_2011", "godaddy",
        "extended_web_traffic_with_missing", "kaggle_web_traffic_weekly",
        "wiki-rolling_nips",
    ],
    "Energy": [
        "australian_electricity_demand", "covid19_energy",
        "elecdemand", "elf",
        "gfc12_load", "gfc14_load", "gfc17_load",
        "largest_2017", "largest_2018", "largest_2019",
        "largest_2020", "largest_2021",
        "lcl", "london_smart_meters_with_missing",
        "residential_load_power", "residential_pv_power",
        "solar_power", "wind_farms_with_missing", "wind_power",
    ],
    "Buildings": [
        "bdg-2_bear", "bdg-2_fox", "bdg-2_panther", "bdg-2_rat",
        "buildings_900k",
    ],
    "Nature/Weather": [
        "beijing_air_quality", "borealis", "china_air_quality",
        "oikolab_weather", "sceaux", "smart",
        "subseasonal", "subseasonal_precip",
        "sunspot_with_missing",
    ],
    "Climate": [
        "cmip6_1850", "cmip6_1855", "cmip6_1860", "cmip6_1865", "cmip6_1870",
        "cmip6_1875", "cmip6_1880", "cmip6_1885", "cmip6_1890", "cmip6_1895",
        "cmip6_1900", "cmip6_1905", "cmip6_1910", "cmip6_1915", "cmip6_1920",
        "cmip6_1925", "cmip6_1930", "cmip6_1935", "cmip6_1940", "cmip6_1945",
        "cmip6_1950", "cmip6_1955", "cmip6_1960", "cmip6_1965", "cmip6_1970",
        "cmip6_1975", "cmip6_1980", "cmip6_1985", "cmip6_1990", "cmip6_1995",
        "cmip6_2000", "cmip6_2005", "cmip6_2010",
        "era5_1989", "era5_1990", "era5_1991", "era5_1992", "era5_1993",
        "era5_1994", "era5_1995", "era5_1996", "era5_1997", "era5_1998",
        "era5_1999", "era5_2000", "era5_2001", "era5_2002", "era5_2003",
        "era5_2004", "era5_2005", "era5_2006", "era5_2007", "era5_2008",
        "era5_2009", "era5_2010", "era5_2011", "era5_2012", "era5_2013",
        "era5_2014", "era5_2015", "era5_2016", "era5_2017", "era5_2018",
    ],
    "Healthcare": [
        "cdc_fluview_ilinet", "cdc_fluview_who_nrevss",
        "covid_mobility", "project_tycho",
    ],
    "Sales": [
        "cockatoo", "favorita_sales", "favorita_transactions", "m5",
    ],
    "Econ/Finance": [
        "bitcoin_with_missing", "bull", "fred_md", "hog", "ideal",
    ],
    "Benchmarks": [
        "m1_monthly", "m1_quarterly", "m1_yearly",
        "monash_m3_monthly", "monash_m3_other",
        "monash_m3_quarterly", "monash_m3_yearly",
        "cif_2016_12", "cif_2016_6",
        "nn5_daily_with_missing", "nn5_weekly",
        "tourism_monthly", "tourism_quarterly", "tourism_yearly",
    ],
    "Other": [
        "kdd2022", "pdb", "spain", "temperature_rain_D_short",
    ],
}

# ===================================================================
# DEFAULT CONFIG
# ===================================================================

DEFAULT_CONFIG = {
    # -- dataset -----------------------------------------------
    "DATASET_NAME": DATASETS["eval"],
    "DATASETS": DATASETS,
    "CONFIGS": EVAL_CONFIGS,                    # friendly-name → HF config
    "ALL_EVAL_CONFIGS": ALL_EVAL_CONFIGS,       # full flat list (97)
    "EVAL_BY_DOMAIN": EVAL_BY_DOMAIN,           # domain → list of configs
    "EVAL_BY_FREQ": EVAL_BY_FREQ,               # freq → list of configs
    "ALL_PRETRAIN_SUBSETS": ALL_PRETRAIN_SUBSETS,  # full flat list (143)
    "PRETRAIN_BY_DOMAIN": PRETRAIN_BY_DOMAIN,   # domain → list of subsets

    # -- sampling ----------------------------------------------
    "N_SAMPLES_TO_LOAD": 10.000,
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

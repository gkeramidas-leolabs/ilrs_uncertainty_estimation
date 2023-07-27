import asyncio
import sys
from pathlib import Path

modules_path = Path("/Users/gkeramidas/Projects/ilrs_uncertainty_estimation")
sys.path.append(str(modules_path))

import pandas as pd
import orekit
from orekit.pyhelpers import setup_orekit_curdir
import validation_utilities as vu

# Orekit setup
orekit_vm = orekit.initVM()
setup_orekit_curdir(
    "/Users/gkeramidas/Projects/ilrs_uncertainty_estimation/leolabs-config-data-dynamic/"
)

# Reading of dataframes
df = pd.read_csv("/Users/gkeramidas/2023-07-15-residuals.csv")
scale_df = pd.read_csv("/Users/gkeramidas/2023-07-15-scale-factors.csv")

# Turn column entries to datetime objects
scale_df["created_at"] = pd.to_datetime(scale_df["created_at"])

# Small dataframe for testing
df1 = df.iloc[0:4]

# API calls for propagations
props_df = asyncio.run(vu.create_propagations_df(df1))
props_df.head()

# Join propagations with original column
df1 = pd.concat([df1, props_df], axis=1)

# Calculating descaled covariance and joining to the dataframe
descaled_column = df1.apply(vu.add_descaled_column, args=(scale_df,), axis=1)
df1 = pd.concat([df1, descaled_column], axis=1)

# Calculate radar coordinates in EME2000 and joining to the dataframe
radar_columns_df = df1.apply(vu.radar_coordinates_columns, axis=1)
df1 = pd.concat([df1, radar_columns_df], axis=1)

# Calculate residuals and compare to the original ones/ Optional step just for debugging
calculated_residuals_df = df1.apply(vu.calculate_residuals, axis=1)
print(calculated_residuals_df.head())
print(df1[["range_res", "doppler_res"]].head())

# Project the originally scaled covariance in measurement space and attach it to the dataframe
projected_scaled_cov_df = df1.apply(
    vu.project_scaled_covariance_to_measurements_space, axis=1
)
df1 = pd.concat([df1, projected_scaled_cov_df], axis=1)

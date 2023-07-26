import asyncio
import sys
from pathlib import Path

modules_path = Path("/Users/gkeramidas/Projects/ilrs_uncertainty_estimation")
sys.path.append(str(modules_path))

import pandas as pd
import validation_utilities as vu

df1 = pd.read_csv("/Users/gkeramidas/2023-07-15-residuals.csv")

df1test = df1.iloc[0:4]

props_df = asyncio.run(vu.create_propagations_df(df1test))
props_df.head()

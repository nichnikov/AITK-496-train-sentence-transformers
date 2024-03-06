import os
import pandas as pd

test_df = pd.read_feather(os.path.join("data", "sys_1_test_queries.feather"))
test_df.to_csv(os.path.join("data", "sys_1_test_queries.csv"), sep="\t", index=False)
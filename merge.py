import pandas as pd


combined_csv = pd.concat([pd.read_csv(f) for f in ["results/b28_res_pt0.csv",
                                                   "results/b28_res_pt1.csv"]])
combined_csv.to_csv("b28_res.csv", index=False)

import pandas as pd


combined_csv = pd.concat([pd.read_csv(f) for f in ["output/anaph_only_b28_lr_default_res.part0.csv",
                                                   "output/anaph_only_b28_lr_default_res.part1.csv"]])
combined_csv.to_csv("results/anaph_only_b28_lr_default.csv", index=False)

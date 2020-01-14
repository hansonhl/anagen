import pandas as pd

raw_res_file = "results/b28_res.csv"
model_dir = "/home/hansonlu/links/data/anagen_models/anagen_b28_model"


raw_df = pd.read(raw_res_file)

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)


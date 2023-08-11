import pandas as pd
import random
import numpy as np

dataset_name="mini_filled_synthetic_dataset.csv"

df = pd.read_csv(dataset_name)
df['class'] = df['class'].apply(lambda x: x[:-3])

df.to_csv("mini_clustering_synthetic_dataset.csv")
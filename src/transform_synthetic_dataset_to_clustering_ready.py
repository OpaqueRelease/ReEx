import pandas as pd
import random
import numpy as np

dataset_name="mixed_classes_dataset.csv"

df = pd.read_csv(dataset_name)
df['class'] = df['class'].apply(lambda x: x[:-3])

df.to_csv("mixed_dataset_clustering.csv")
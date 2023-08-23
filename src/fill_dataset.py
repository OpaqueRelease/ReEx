import pandas as pd
import random
import numpy as np

dataset_name="mini_synthetic_dataset.csv"
# features = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19","F20","F21","F22","F23","F24"]
features = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12", "F13", "F14"]


# arr_random = np.random.default_rng().uniform(low=0, high=1, size=(540,24))
arr_random = np.random.default_rng().uniform(low=0, high=1, size=(79,14))

new = pd.DataFrame(arr_random, columns=features)

df = pd.read_csv(dataset_name)

for index, row in df.iterrows():
    for feature in features:
        if row[feature]==1:
            new.at[index, feature] = 42
new['class'] = df['class']

new.to_csv("mini_filled_synthetic_dataset.csv")
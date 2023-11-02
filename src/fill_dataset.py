import pandas as pd
import random
import numpy as np

dataset_name="mixed_dataset_clustering.csv"
features = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19","F20","F21","F22","F23","F24","F25","F26","F27","F28","F29","F30"]
#features = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12", "F13", "F14"]


# arr_random = np.random.default_rng().uniform(low=0, high=1, size=(97,30))

# new = pd.DataFrame(arr_random, columns=features)

df = pd.read_csv(dataset_name)

for index, row in df.iterrows():
    for feature in ["F21","F22","F23","F24","F25","F26","F27","F28","F29","F30"]:
        if row[feature]==1:
            df.at[index, feature] = 1
        else:
            df.at[index, feature] = random.randint(0,1)
# new['class'] = df['class']

df.to_csv("filled_mixed.csv")
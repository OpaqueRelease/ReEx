import pandas as pd 
from operator import itemgetter
import matplotlib.pyplot as plt

df = pd.read_csv('shapleys.csv', sep=',')
shaps = {}
shap_n = {}

for column in df:
    sum = 0
    val = 0
    print(column)
    for value in df[column].values:
        if value > 0:
            sum += value
            val += 1
    shaps[column] = sum
    shap_n[column] = val

print(shaps)
print(shap_n)

for feature in shaps: #normalize
    if shap_n[feature] > 0:
        shaps[feature] = shaps[feature] / shap_n[feature]

res = dict(sorted(shaps.items(), key = itemgetter(1), reverse = True)[:11])
 
# printing result
print("The top N value pairs are  " + str(res))
del res['Unnamed: 0']

names = list(res.keys())
values = list(res.values())

plt.bar(range(len(res)), values, tick_label=names)
plt.xlabel("Features")
plt.ylabel("Feature importance")
plt.show()
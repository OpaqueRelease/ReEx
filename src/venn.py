import json
import pandas as pd
basic_file = "../results/bbc_wordnet.json"
target_file = "../results/represent.json"

f = open(basic_file)
data_basic = json.load(f)
data_basic = json.loads(data_basic)

target_f = open(target_file)
data_target = json.load(target_f)
data_target = json.loads(data_target)

for key in data_basic["resulting_generalization"]:
    if key == "average_association" or key == "average_depth":
        continue 
    common = list(set(data_basic["resulting_generalization"][key]["terms"]).intersection(set(data_target["resulting_generalization"][key]["terms"])))
    basic = list(set(data_basic["resulting_generalization"][key]["terms"]).difference(set(data_target["resulting_generalization"][key]["terms"])))
    target = list(set(data_target["resulting_generalization"][key]["terms"]).difference(set(data_basic["resulting_generalization"][key]["terms"])))
    frame = pd.DataFrame({'Ungeneralized': pd.Series(common), 'Generalized': pd.Series(basic), 'Generalizations': pd.Series(target)})
    print(frame)
    frame.to_csv(key+".csv")
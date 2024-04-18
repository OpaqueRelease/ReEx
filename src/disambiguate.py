from nltk.wsd import lesk
import pandas as pd
from nltk.corpus import wordnet


df = pd.read_csv("bbc.tsv", sep='\t', header=0, index_col = False)

for index, row in df.iterrows():
    sentence = row["data"]
    stringer = ""
    sentence_split = sentence.split()
    for word in sentence_split:
        feature = lesk(sentence_split, word, synsets=wordnet.synsets(word, lang='eng'))
        if feature:
            stringer += feature.name().replace(".", "__") + " "

    df.at[index, 'data'] = stringer

df.to_csv("bbc_disambiguated.tsv", sep='\t')

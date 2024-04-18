## parsers and such.
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import pandas as pd
import time
import gzip
import networkx as nx
import obonet
import timeit
from collections import defaultdict
import sys
import os
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from textblob import Word
from goatools import obo_parser
import wget
import os


def read_generic_gaf(gaf_file):
    """
    Reads .gaf mapping file
    """
    symmap = defaultdict(set)
    with gzip.open(gaf_file,"rt") as gf:
        for line in gf:
            line = line.strip().split("\t")
            if "UniProt" in line[0] and len(line) > 5:
                # print(line[4], line[9])
                symmap[line[4]].add(line[9])
    return symmap

go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
data_folder = os.getcwd() + '/data'

# Check if we have the ./data directory already
if(not os.path.isfile(data_folder)):
    # Emulate mkdir -p (no error if folder exists)
    try:
        os.mkdir(data_folder)
    except OSError as e:
        if(e.errno != 17):
            raise e
else:
    raise Exception('Data path (' + data_folder + ') exists as a file. '
                   'Please rename, remove or change the desired location of the data path.')

# Check if the file exists already
if(not os.path.isfile(data_folder+'/go-basic.obo')):
    go_obo = wget.download(go_obo_url, data_folder+'/go-basic.obo')
else:
    go_obo = data_folder+'/go-basic.obo'

go = obo_parser.GODag(go_obo)
# print(go)


fn = "LUAD-1.csv"
#mapping = read_generic_gaf('../example/mapping/goa_human.gaf.gz')
#print(mapping)
df = pd.read_csv(fn)
# del df['index']
print(df)

for index, row in df.iterrows():
    if not pd.isna(row['Ungeneralized']) and row['Ungeneralized'] in go:
        df.at[index, 'Ungeneralized'] = str(go[row['Ungeneralized']].name)
    if not pd.isna(row['Generalized']) and row['Generalized'] in go:
    
        df.at[index,'Generalized'] = str(go[row['Generalized']].name)
    if not pd.isna(row['Generalizations']) and row['Generalizations'] in go:
        print(go[row['Generalizations']].name)
        
        df.at[index, 'Generalizations'] = str(go[row['Generalizations']].name)
print(df)
df.to_csv("mod_"+fn)
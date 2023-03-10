import os
import json 
import torch
import clip
from PIL import Image
import sng_parser
from tqdm import tqdm 
import codecs
import numpy as np



json_path = '/data/mshukor/data/recipe1m/recipe1m_13m/layer1.json'
data1 = json.load(open(json_path,'r'))

json_path = '/data/mshukor/data/recipe1m/recipe1m_13m/layer2+.json'
data_2 = json.load(open(json_path,'r'))

ids_with_images = []

for d in tqdm(data_2):
    id_ = d['id']
    if len(d['images']) > 0:
        ids_with_images.append(id_)


new_ids = {'train': [], 'test': [], 'val': []}


for d in tqdm(data1):
    id_ = d['id']
    split = d['partition']
    if id_ in ids_with_images:
        new_ids[split].append(id_)


output_path = '/data/mshukor/data/recipe1m/recipe1m_13m/original_ids.json'
with open(output_path, 'w') as f:
    json.dump(new_ids, f)
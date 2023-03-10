import os
import json 
import torch
import clip
from PIL import Image
from tqdm import tqdm 
import codecs
import numpy as np

import torchvision.transforms as transforms


def default_image_tf(scale_size, crop_size, mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]  # resnet imagenet
    std = std or [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),  # divide by 255 automatically
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


image_root = '/gpfsdswork/dataset/Recipe1M+/images_recipe1M+'
new_data_2 = []
transform = default_image_tf(256, 224) #transforms.ToTensor() #default_image_tf(256, 224)

json_path = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/recipe1m_13m/layer2+.json'
data_2 = json.load(open(json_path,'r'))


for i, image_entry in tqdm(enumerate(data_2)):
    item = {'id': image_entry['id'], 'images': []}
    for img_name in image_entry['images']:
        name = img_name['id']

        name = '/'.join(name[:4])+'/'+name

        img = Image.open(os.path.join(image_root, name))
        try:
            img = img.convert('RGB')
            img = transform(img)
            item['images'].append(img_name)
            
        except RuntimeError:
            print(i, img_name)
            pass
    new_data_2.append(item)



output_path = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/recipe1m_13m/filtered_layer2+.json'
with open(output_path, 'w') as f:
    json.dump(new_data_2, f)
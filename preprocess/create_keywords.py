import os
import json 
import torch
import clip
from PIL import Image
import sng_parser
from tqdm import tqdm 
from statistics import median
import numpy as np
import codecs
import PIL

from nltk.stem import PorterStemmer


import sys 
sys.path.append("/home/mshukor/tfood") 

from torch import nn, optim
from torchvision import models, datasets, transforms
import torchvision

from classification.data import ImageFolder_Context
from torch.utils.data import DataLoader

import torch
from torchvision.datasets.folder import ImageFolder, default_loader


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFolder_custom(ImageFolder):
    def __init__(
        self,
        root,
        transform,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        return_path=False,):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.return_path = return_path

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_path:
            return sample, target, path
        else:
            return sample, target


def parse_nouns_attr_relations(text, extract_rel=True, extract_att=True):
    graph = sng_parser.parse(text)
    # parse entities
    obj = []
    obj_att = []
    rel = []
    entities = graph['entities']
    
    
    for o in entities:
        obj.append(o['head'])
        if extract_att:
            for mod in o['modifiers']:
                if mod['dep'] != 'det':
                    obj_att.append((o['head'],  mod['span']))
    if extract_rel:
        for r in graph['relations']:
            sub = entities[r['subject']]['head']
            re = r['relation']
            ob = entities[r['object']]['head']
            rel.append((sub, re, ob))
                
    return obj, obj_att, rel



def dict_sort(d):
    mean = sum(d.values()) / len(d)
    med = median(d.values())
    sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    return sorted_d, mean, med
    
def extract_titles(json_path, output_path):
    with open(json_path, 'r') as f:
        layer1 = json.load(f)

    titles = []

    for i, d in tqdm(enumerate(layer1)):
        title = d['title'].lower()
        titles.append(title)

    with open(output_path, 'w') as f:
        json.dump(titles, f)
    return titles


def extract_keywords(json_path, extract_rel=False, extract_att=False, 
    output_path='/data/mshukor/data/our_albef_data', max_num_keys=None, thresh=None, nlvr2=False, key='ingredients'):
    

    with open(json_path, 'r') as f:
        layer1 = json.load(f)
    layer1_ = {data['id']:data for data in tqdm(layer1)}

    print('finish reading')
    text = set()
    filter_titles = ['-', '_', '/']
    objs = dict()
    atts = dict()
    rels = dict()

    ps = PorterStemmer()

    print(len(layer1_))
    for i, (k, v) in tqdm(enumerate(layer1_.items())):
        total = []
        if key == 'title' or key == 'all':
            title = v['title'].lower()
            total+=[title]
        if key == 'ingredients' or key == 'all': 
            ingrs = [ing['text'].lower() for ing in v['ingredients']]
            total+=ingrs
        if key == 'instructions' or key == 'all': 
            insts = [inst['text'].lower() for inst in v['instructions']]
            total+=insts

        for txt in total:
            for f in filter_titles:
                txt = txt.replace(f, ' ')


            objects, objects_attributes, relations = parse_nouns_attr_relations(txt, extract_rel=extract_rel, extract_att=extract_att)
            objects = [ps.stem(t.lower()) for t in objects]
            for o in objects:
                if len(o) > 2:
                    if o in objs:
                        objs[o] += 1
                    else:
                        objs[o] = 0
                
            if extract_att:
                for o_a in objects_attributes:
                    tmp = o_a[0]+' '+o_a[1]
                    if tmp in atts:
                        atts[tmp] += 1
                    else:
                        atts[tmp] = 0
                                    
            if extract_rel:
                for r in relations:
                    tmp = r[0]+' '+r[1]+' '+r[2]
                    if tmp in rels:
                        rels[tmp] += 1
                    else:
                        rels[tmp] = 0
        # if i > 2000:
        #     break
            
    objs, mean_objs, med_objs = dict_sort(objs)
    print(len(objs), mean_objs, med_objs)
    
    if max_num_keys is not None:
        new_objs = list(objs.keys())[:max_num_keys]
    elif thresh is not None:
        new_objs = [o[0] for o in objs.items() if o[1] > thresh]
    else:
        new_objs = objs
        
    with open(output_path, 'w') as f:
        json.dump(new_objs, f)
    print('After filtering', len(new_objs))
    
    if extract_att:
        atts, mean_atts, med_atts = dict_sort(atts)
    if extract_rel:
        rels, mean_rels, med_rels = dict_sort(rels)
    
    return new_objs, atts, rels




def save_clip_embeddings(json_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    data_ = json.load(open(json_path,'r'))
    text_embed = dict()
    with torch.no_grad():
        for i, t in tqdm(enumerate(data_)):
            
            try:
                text_tokens = clip.tokenize(t).to(device)
                text_features = model.encode_text(text_tokens)
            except RuntimeError:
                print(t)
                continue
            text_embed[t] = text_features.cpu().numpy().tolist()

    json.dump(text_embed, codecs.open(output_path, 'w', encoding='utf-8'))
    
    return text_embed


def dict_to_tensor(dict_data):
    embeds = []
    index2kword = dict()
    for i, (k, d) in tqdm(enumerate(dict_data.items())):
        embeds.append(torch.from_numpy(np.array(d)))
        index2kword[i] = k 
        
    embeds = torch.cat(embeds, dim=0)
    return embeds, index2kword


def select_topk(image_features, text_features, k=10):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = image_features @ text_features.t()
    
    top_k_indices = torch.topk(logits_per_image, k, dim=-1)[1]
    
    return logits_per_image, top_k_indices

def create_clip_Da_dataset(json_path, embeddings_path, k=10, data_dir=None, clip_path=None, max_idx=None, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if clip_path is None:
        model, preprocess = clip.load("ViT-B/16", device=device)
    else:
        model, preprocess = clip.load(clip_path, device=device)
        
    print(json_path)
    data_ = json.load(open(json_path,'r'))

    data_embed = json.load(open(embeddings_path,'r'))
    embeddings, index2kword = dict_to_tensor(data_embed)
    embeddings  = embeddings.to(device).type(torch.float16)
    
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            if 'kwords' not in d:
                image_path = d['image']
                if data_dir is not None:
                    image_path = os.path.join(data_dir, ('/').join(image_path.split('/')[4:]))
                elif image_root is not None:
                    image_path = os.path.join(image_root, image_path)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)

                logits_per_image, top_k_indices = select_topk(image_features, embeddings, k=k)

                topk = top_k_indices[0].cpu().numpy().tolist()

                kwords = [index2kword[i] for i in topk]

                d['kwords'] = kwords
                
                if (i + 1) % 500000 == 0:
                    with open(output_path, 'w') as file:
                        json.dump(data_, file)
            if max_idx is not None and i > max_idx:
                break
            
    with open(output_path, 'w') as file:
        json.dump(data_, file)
        
    return data_



        
def create_titles(json_path, output_path='/data/mshukor/data/our_albef_data'):
    data_ = json.load(open(json_path,'r'))
    num_no_objects = 0
    for i, d in tqdm(enumerate(data_)):
        if isinstance(d['caption'], list):
            cap = d['caption'][0]
        else:
            cap = d['caption']
        objects, objects_attributes, relations = parse_nouns_attr_relations(cap, extract_rel=False, extract_att=False)
        if len(objects) > 0:
            title = (' and ').join(objects)
            d['title'] = title
        else:
            d['title'] = d['caption']
            num_no_objects+=1

        
            
    print('number of captions wihtout objects:', num_no_objects)    
    with open(output_path, 'w') as f:
        json.dump(data_, f)
    
    
    return data_


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import sys
sys.path.append("/home/mshukor/tfood/")


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def save_image_embeddings(dir_data, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', 
    image_root=None, split='train'):
    from recipe1m.datasets.recipe1m import Images 

    image_tf = _transform(224) #utils.default_image_tf(256, 224)
    image_from = 'database'
    images_dataset = Images(dir_data, split, 1, 4, image_from=image_from, image_tf=image_tf, get_all_images=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    # data_ = json.load(open(json_path,'r'))
   
    image_embed = dict()
    num_corrupted = 0
    print("number of samples:", len(images_dataset))
    with torch.no_grad():
        try:
            for i, samples in tqdm(enumerate(images_dataset)):
                for d in samples['samples']:
                    image_data, index_img, path_img = d['data'], d['index'], d['path']

                    if path_img not in image_embed:

                        image = image_data.unsqueeze(0).to(device) #preprocess(Image.open(image_path)).unsqueeze(0).to(device)

                        image_features = model.encode_image(image)

                        image_embed[path_img] = image_features.cpu().numpy().tolist()
        except TypeError:
            num_corrupted+=1
            print(num_corrupted)
            print(path_img)
            pass                    
    print('size of dict', len(image_embed))
    # with open(output_path, 'w') as f:
    #     json.dump(image_embed, f)

    json.dump(image_embed, codecs.open(output_path, 'w', encoding='utf-8'))
        
    return image_embed


def save_image_embeddings_recipe1m_13m(path_ids, path_image_json, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', 
    image_root=None, split='train', transform=None):


    ids = json.load(open(path_ids,'r'))
    ids = ids[split]


    layer2_ = json.load(open(path_image_json,'r'))
    layer2 = {}
    for d in layer2_:
        layer2[d['id']] = d['images'] 


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    image_tf = _transform(224)

    image_embed = dict()
    num_corrupted = 0
    print("number of samples:", len(layer2))
    with torch.no_grad():
        try:
            for i, (image_id, image_entry) in tqdm(enumerate(layer2.items())):
                if image_id not in image_embed:
                    if len(image_entry) > 0:
                        img_name = image_entry[0]
                        img_name = img_name['id']

                        img_name = '/'.join(img_name[:4])+'/'+img_name
                        img = Image.open(os.path.join(image_root, img_name)).convert('RGB')
                        
                        img = image_tf(img)

                        
                        image = img.unsqueeze(0).to(device) #preprocess(Image.open(image_path)).unsqueeze(0).to(device)

                        image_features = model.encode_image(image)

                        image_embed[image_id] = image_features.cpu().numpy().tolist()
        except TypeError:
            num_corrupted+=1
            print(num_corrupted)
            print(image_id)
            pass                    
    print('size of dict', len(image_embed))
    # with open(output_path, 'w') as f:
    #     json.dump(image_embed, f)

    json.dump(image_embed, codecs.open(output_path, 'w', encoding='utf-8'))
        
    return image_embed



def save_image_embeddings_food101(data_dir, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', 
    image_root=None, split='train', transform=None):

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data = ImageFolder_Context(data_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), return_path=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    image_embed = dict()
    num_corrupted = 0
    print("number of samples:", len(data))
    with torch.no_grad():
        try:
            for i, (image, target, path) in tqdm(enumerate(DataLoader(data, batch_size=1))):
                img_id = path[0].split('/')[-1].split('.')[0]
                if img_id not in image_embed:

                    image_features = model.encode_image(image.to(device))
                    image_embed[img_id] = image_features.cpu().numpy().tolist()


        except TypeError:
            num_corrupted+=1
            print(num_corrupted)
            print(path)
            pass                    
    print('size of dict', len(image_embed))


    json.dump(image_embed, codecs.open(output_path, 'w', encoding='utf-8'))
        
    return image_embed



def save_image_embeddings_imagenet(dir_data, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/'):


    image_tf = _transform(224)
    dataset = ImageFolder_custom(dir_data, transform=image_tf, return_path=True)

    data = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    # data_ = json.load(open(json_path,'r'))
   
    image_embed = dict()
    num_corrupted = 0
    print("number of samples:", len(data))
    with torch.no_grad():
        try:
            for i, (image, target, path) in tqdm(enumerate(data)):

                img_id = path[0].split('/')[-1].split('.')[0]

                if img_id not in image_embed:

                    image = image.to(device) #preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    image_features = model.encode_image(image)
                    image_embed[img_id] = image_features.cpu().numpy().tolist()
        except TypeError:
            num_corrupted+=1
            print(num_corrupted)
            print(img_id)
            pass                    
    print('size of dict', len(image_embed))
    # with open(output_path, 'w') as f:
    #     json.dump(image_embed, f)

    json.dump(image_embed, codecs.open(output_path, 'w', encoding='utf-8'))
        
    return image_embed


def create_clip_Da_dataset_from_saved_imagenet(data_dir, embeddings_path, image_embeddings_path, k=10, 
    output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', split='train'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    # data_ = json.load(open(json_path,'r'))

    data_embed = json.load(open(embeddings_path,'r'))
    embeddings, index2kword = dict_to_tensor(data_embed)
    embeddings  = embeddings.to(device).type(torch.float16)
    
    data_embed_images = json.load(open(image_embeddings_path,'r')) #(name, image)
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)

    image_to_kw = dict()

    image_tf = _transform(224)
    dataset = ImageFolder_custom(data_dir, transform=image_tf, return_path=True)

    data = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=4,
        )


    corrupted = 0

    print(len(data_embed_images), len(index2image_path), len(image_path2index), image_embeddings.shape)
    with torch.no_grad():
        try:
            for i, (image, target, path) in tqdm(enumerate(data)):
                img_id = path[0].split('/')[-1].split('.')[0]

                if img_id not in image_to_kw:

                    try:
                        image_features = image_embeddings[image_path2index[img_id]].unsqueeze(0)
                    except:
                        print('not found', img_id)
                        corrupted+=1
                        print('corrupted', corrupted)
                        continue

                    logits_per_image, top_k_indices = select_topk(image_features, embeddings, k=k)
                    topk = top_k_indices[0].cpu().numpy().tolist()

                    kwords = [index2kword[i] for i in topk]

                    image_to_kw[img_id] = kwords

        except TypeError:
            pass                    



    print('dataset new size', len(image_to_kw))
    with open(output_path, 'w') as file:
        json.dump(image_to_kw, file)
        
    return image_to_kw

def create_clip_Da_dataset_from_saved_food101(data_dir, embeddings_path, image_embeddings_path, k=10, 
    output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', split='train'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    # data_ = json.load(open(json_path,'r'))

    data_embed = json.load(open(embeddings_path,'r'))
    embeddings, index2kword = dict_to_tensor(data_embed)
    embeddings  = embeddings.to(device).type(torch.float16)
    
    data_embed_images = json.load(open(image_embeddings_path,'r')) #(name, image)
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)

    image_to_kw = dict()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data = ImageFolder_Context(data_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), return_path=True)


    corrupted = 0

    print(len(data_embed_images), len(index2image_path), len(image_path2index), image_embeddings.shape)
    with torch.no_grad():
        try:
            for i, (image, target, path) in tqdm(enumerate(DataLoader(data, batch_size=1))):
                img_id = path[0].split('/')[-1].split('.')[0]

                if img_id not in image_to_kw:

                    try:
                        image_features = image_embeddings[image_path2index[img_id]].unsqueeze(0)
                    except:
                        print('not found', img_id)
                        corrupted+=1
                        print('corrupted', corrupted)
                        continue

                    logits_per_image, top_k_indices = select_topk(image_features, embeddings, k=k)
                    topk = top_k_indices[0].cpu().numpy().tolist()

                    kwords = [index2kword[i] for i in topk]

                    image_to_kw[img_id] = kwords

        except TypeError:
            pass                    



    print('dataset new size', len(image_to_kw))
    with open(output_path, 'w') as file:
        json.dump(image_to_kw, file)
        
    return image_to_kw


def create_clip_Da_dataset_from_saved_recipe1m_13m(path_ids, path_image_json, embeddings_path, image_embeddings_path, k=10, 
    output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', split='train'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    # data_ = json.load(open(json_path,'r'))

    data_embed = json.load(open(embeddings_path,'r'))
    embeddings, index2kword = dict_to_tensor(data_embed)
    embeddings  = embeddings.to(device).type(torch.float16)
    
    data_embed_images = json.load(open(image_embeddings_path,'r')) #(name, image)
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)

    image_to_kw = dict()

    ids = json.load(open(path_ids,'r'))
    ids = ids[split]


    layer2_ = json.load(open(path_image_json,'r'))
    layer2 = {}
    for d in layer2_:
        layer2[d['id']] = d['images'] 


    corrupted = 0

    print(len(data_embed_images), len(index2image_path), len(image_path2index), image_embeddings.shape)
    with torch.no_grad():
        try:
            for i, image_id in tqdm(enumerate(ids)):
                image_entry = layer2[image_id]
                if len(image_entry) > 0:
                    img_name = image_entry[0]

                    if image_id not in image_to_kw:

                        try:
                            image_features = image_embeddings[image_path2index[image_id]].unsqueeze(0)
                        except:
                            print('not found', image_id)
                            corrupted+=1
                            print('corrupted', corrupted)
                            continue

                        logits_per_image, top_k_indices = select_topk(image_features, embeddings, k=k)
                        topk = top_k_indices[0].cpu().numpy().tolist()

                        kwords = [index2kword[i] for i in topk]

                        image_to_kw[image_id] = kwords

        except TypeError:
            pass                    



    print('dataset new size', len(image_to_kw))
    with open(output_path, 'w') as file:
        json.dump(image_to_kw, file)
        
    return image_to_kw


def create_clip_Da_dataset_from_saved(dir_data, embeddings_path, image_embeddings_path, k=10, 
    output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', split='train'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    # data_ = json.load(open(json_path,'r'))

    data_embed = json.load(open(embeddings_path,'r'))
    embeddings, index2kword = dict_to_tensor(data_embed)
    embeddings  = embeddings.to(device).type(torch.float16)
    
    data_embed_images = json.load(open(image_embeddings_path,'r')) #(name, image)
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)

    image_to_kw = dict()

    from recipe1m.datasets.recipe1m import Images 

    image_tf = _transform(224) #utils.default_image_tf(256, 224)
    image_from = 'database'
    images_dataset = Images(dir_data, split, 1, 4, image_from=image_from, image_tf=image_tf, get_all_images=True)
    corrupted = 0

    print(len(data_embed_images), len(index2image_path), len(image_path2index), image_embeddings.shape)
    with torch.no_grad():
        try:
            for i, samples in tqdm(enumerate(images_dataset)):
                for d in samples['samples']:
                    image_data, index_img, path_img = d['data'], d['index'], d['path']

                    if path_img not in image_to_kw:

                        image = image_data.unsqueeze(0).to(device) #preprocess(Image.open(image_path)).unsqueeze(0).to(device)

                        # image_features = model.encode_image(image)

                        try:
                            image_features = image_embeddings[image_path2index[path_img]].unsqueeze(0)
                        except:
                            print('not found', path_img)
                            corrupted+=1
                            print('corrupted', corrupted)
                            continue

                        logits_per_image, top_k_indices = select_topk(image_features, embeddings, k=k)
                        topk = top_k_indices[0].cpu().numpy().tolist()

                        kwords = [index2kword[i] for i in topk]

                        image_to_kw[path_img] = kwords

        except TypeError:
            pass                    



    print('dataset new size', len(image_to_kw))
    with open(output_path, 'w') as file:
        json.dump(image_to_kw, file)
        
    return image_to_kw



def compute_sim(image_features, text_features):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = image_features @ text_features.t()
    return logits_per_image

def save_captions_embeddings(json_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)
    data_ = json.load(open(json_path,'r'))
    text_embed = dict()
    with torch.no_grad():
        for i, t in tqdm(enumerate(data_)):
            text = t['caption']
            text_tokens = clip.tokenize(text, truncate=True).to(device)
            text_features = model.encode_text(text_tokens)
            text_embed[text] = text_features.cpu().numpy().tolist()
    print('saving...')
    json.dump(text_embed, codecs.open(output_path, 'w', encoding='utf-8'))
    
    return text_embed



def save_mini_json(json_path, output_path, size=10000):
    data_ = json.load(open(json_path,'r'))
    mini_data = []
    for i, d in enumerate(tqdm(data_)):
        mini_data.append(d)
        if i > size:
            break
    json.dump(mini_data, codecs.open(output_path, 'w', encoding='utf-8'))      
    
    return mini_data


# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in tqdm(range(0, len(l), n)):
#         yield l[i:i + n]

def filter_topk_dataset_from_saved_sim(json_path, per=1, 
    output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/',):


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))


    new_data = sorted(data_, key=lambda d: d['sim'], reverse=True) 
    
    num_items = int(per*len(new_data))
    print("number of items remaining:", num_items)
    filtered_data = new_data[:num_items]
    with open(output_path, 'w') as file:
        json.dump(filtered_data, file)

        
    return filtered_data

def filter_topk_dataset(json_path, per=1, 
    output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, 
    overwrite=True, output_path_orig=None, save_original=False, batch_size=8):


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))

    # data_ = data_[:2000]
    num_corrupted = 0
    with torch.no_grad():
        for idx in tqdm(range(0, len(data_), batch_size)):
            batch = data_[idx:idx + batch_size]
        
            images = []
            captions = []
            for i, d in enumerate(batch):

                image_name = d['image']
                
                if image_root is not None:
                    image_path = os.path.join(image_root, image_name)
                else:
                    image_path = image_name

                image = preprocess(Image.open(image_path)).unsqueeze(0)

                caption = d['caption']
                text_tokens = clip.tokenize(caption, truncate=True)

                images.append(image)
                captions.append(text_tokens)

            images = torch.cat(images, dim=0).to(device)
            captions = torch.cat(captions, dim=0).to(device)

            image_features = model.encode_image(images)
            caption_features = model.encode_text(captions)

            for i, d in enumerate(batch):
                sim = compute_sim(image_features[i].unsqueeze(0), caption_features[i].unsqueeze(0)).item()
                d['sim'] = sim

            

    new_data = sorted(data_, key=lambda d: d['sim'], reverse=True) 
    
    num_items = int(per*len(new_data))
    print("number of items remaining:", num_items)
    filtered_data = new_data[:num_items]
    with open(output_path, 'w') as file:
        json.dump(filtered_data, file)

    if save_original:
        with open(output_path_orig, 'w') as file:
            json.dump(data_, file)

        
    return filtered_data


def filter_topk_per_image_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=1, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', 
                                             image_root=None, overwrite=True, caption_embed=None, data_=None):

    device = "cpu"

    if data_ is None:
        data_ = json.load(open(json_path,'r'))
    else:
        print('skip reading data')

    if caption_embed is None:
        caption_embed = json.load(open(caption_embeddings_path,'r'))
    else:
        print('skip reading caption embeddings')
        
    caption_embeddings, index2caption = dict_to_tensor(caption_embed)
    caption2index = {v:k for (k, v) in index2caption.items()}
    caption_embeddings  = caption_embeddings.to(device).type(torch.float16)

    data_embed_images = json.load(open(image_embeddings_path,'r'))
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)
    
    data_dict = dict()
        
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            image_path = d['image']
            if image_root is not None:
                image_path = os.path.join(image_root, image_path)

            
            image_features = image_embeddings[image_path2index[image_path]].unsqueeze(0)
            
            caption = d['caption']
            caption_features = caption_embeddings[caption2index[caption]].unsqueeze(0)
            
            
            sim = compute_sim(image_features, caption_features).item()
            
            d['sim'] = sim
            
            if d['image'] in data_dict:
                data_dict[d['image']].append(d)
            else:
                data_dict[d['image']] = [d]

    new_data = []
    for k, ds in data_dict.items():
        new_ds = sorted(ds, key=lambda d: d['sim'], reverse=True) 
        num_items = int(per*len(new_ds))
        new_data += new_ds[:num_items]
        
    with open(output_path, 'w') as file:
        json.dump(new_data, file)
        
    print("new dataset size:", len(new_data) , "before:", len(data_))
    return new_data

def create_clip_captions_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, k=5, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))

    caption_embed = json.load(open(caption_embeddings_path,'r'))
    caption_embeddings, index2caption = dict_to_tensor(caption_embed)
    # caption2index = {v:k for (k, v) in index2caption.items()}
    caption_embeddings  = caption_embeddings.to(device).type(torch.float32)
    
    data_embed_images = json.load(open(image_embeddings_path,'r'))
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float32)
    img_captions = {}
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            image_path = d['image']
            if image_path not in img_captions:
                if image_root is not None:
                    image_path = os.path.join(image_root, image_path)
                    
                # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                # image_features = model.encode_image(image)
                try:
                    image_features = image_embeddings[image_path2index[image_path]].unsqueeze(0)
                except:
                    print('not found', image_path2index[image_path])
                    continue
                logits_per_image, top_k_indices = select_topk(image_features, caption_embeddings, k=k)
                topk = top_k_indices[0].cpu().numpy().tolist()

                kwords = [index2caption[i] for i in topk]

                d['additional_captions'] = kwords

                img_captions[image_path] = kwords
            else:
                d['additional_captions'] = img_captions[image_path]

            

    print('dataset new size', len(data_))
    with open(output_path, 'w') as file:
        json.dump(data_, file)
        
    return data_

def filter_topk_dataset_from_saved(json_path, caption_embeddings_path, image_embeddings_path, per=1, output_path='/data/mshukor/data/our_albef_data/clip_da/json_pretrain/', image_root=None, overwrite=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/16", device=device)

    data_ = json.load(open(json_path,'r'))

    caption_embed = json.load(open(caption_embeddings_path,'r'))
    caption_embeddings, index2caption = dict_to_tensor(caption_embed)
    caption2index = {v:k for (k, v) in index2caption.items()}
    caption_embeddings  = caption_embeddings.to(device).type(torch.float16)
    
    data_embed_images = json.load(open(image_embeddings_path,'r'))
    image_embeddings, index2image_path = dict_to_tensor(data_embed_images)
    image_path2index = {v:k for (k, v) in index2image_path.items()}
    image_embeddings  = image_embeddings.to(device).type(torch.float16)
    
    
    with torch.no_grad():
        for i, d in tqdm(enumerate(data_)):
            image_path = d['image']
            if image_root is not None:
                image_path = os.path.join(image_root, image_path)

            try:
                image_features = image_embeddings[image_path2index[image_path]].unsqueeze(0)
            except:
                continue
            
            caption = d['caption']
            caption_features = caption_embeddings[caption2index[caption]].unsqueeze(0)
            
            
            sim = compute_sim(image_features, caption_features).item()
            
            d['sim'] = sim

            

    new_data = sorted(data_, key=lambda d: d['sim'], reverse=True) 
    
    num_items = int(per*len(new_data))
    print("number of items remaining:", num_items)
    filtered_data = new_data[:num_items]
    with open(output_path, 'w') as file:
        json.dump(filtered_data, file)
        
    return filtered_data
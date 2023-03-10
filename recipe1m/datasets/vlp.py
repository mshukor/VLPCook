import os
import torch
import torch.utils.data as data

from PIL import Image

from bootstrap.lib.options import Options
from bootstrap.datasets import transforms

from bootstrap.lib.options import Options
 
import json
from torch.utils.data import Dataset
import re
import random

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


def default_items_tf():
    return transforms.Compose([
        transforms.ListDictsToDictLists(),
        transforms.PadTensors(value=0),
        transforms.StackTensors()
    ])
 

class Dataset(data.Dataset):

    def __init__(self, dir_data, split, batch_size, nb_threads, items_tf=default_items_tf):
        super(Dataset, self).__init__()
        self.dir_data = dir_data
        self.split = split
        self.batch_size = batch_size
        self.nb_threads = nb_threads
        self.items_tf = default_items_tf

    def make_batch_loader(self, shuffle=True):
        # allways shuffle even for valset/testset
        # see testing procedure

        if Options()['dataset'].get("debug", False):
            return data.DataLoader(self,
                batch_size=self.batch_size,
                num_workers=self.nb_threads,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.items_tf(),
                drop_last=True) # Removing last batch if not full (quick fix accuracy calculation with class 0 only)
        else:
            return data.DataLoader(self,
                batch_size=self.batch_size,
                num_workers=self.nb_threads,
                shuffle=shuffle,
                pin_memory=True,
                collate_fn=self.items_tf(),
                drop_last=True) # Removing last batch if not full (quick fix accuracy calculation with class 0 only)




class RecipeVLP(Dataset):
    def __init__(self, ann_file, transform, max_words=30, data_dir='/data/mshukor/data', tokenizer=None, 
        nb_threads=4, batch_size=100, items_tf=default_items_tf, use_tags=False, 
        only_captions=False, bert=False, use_vcs=False, randkw_p=None, aux_kwords=False, randkw_p_aux=None):        
        self.ann = []
        for f in ann_file:
            tmp =  json.load(open(f,'r'))
            self.ann += tmp
            print('size of', f, len(tmp))
        print(len(self.ann))

        self.transform = transform
        self.max_words = max_words
        for e in self.ann:
            e['image'] = os.path.join(data_dir, ('/').join(e['image'].split('/')[4:]))
            
        self.tokenizer = tokenizer

        self.nb_threads = nb_threads
        self.batch_size = batch_size
        self.split = 'pretrain'
        self.items_tf = items_tf

        self.use_tags = use_tags
        self.num_kws = 15
        self.only_captions = only_captions

        print('use_tags:', use_tags)
        print('only_captions:', only_captions)

        self.bert = bert

        self.use_vcs = use_vcs
        self.randkw_p = randkw_p
        self.aux_kwords = aux_kwords
        print('aux_kwords', aux_kwords)
        
    def __len__(self):
        return len(self.ann)
    

    def make_batch_loader(self, shuffle=True):

        if Options()['dataset'].get("debug", False):
            return data.DataLoader(self,
                batch_size=self.batch_size,
                num_workers=self.nb_threads,
                shuffle=False,
                pin_memory=True,
                collate_fn=self.items_tf(),
                drop_last=True) # Removing last batch if not full (quick fix accuracy calculation with class 0 only)
        else:
            return data.DataLoader(self,
                batch_size=self.batch_size,
                num_workers=self.nb_threads,
                shuffle=shuffle,
                pin_memory=True,
                collate_fn=self.items_tf(),
                drop_last=True) # Removing last batch if not full (quick fix accuracy calculation with class 0 only)
 

    def __getitem__(self, index):    
        item = {}
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        
        title = ann['title']

        if self.use_tags:
            kws = list(set(ann['objects']))[:self.num_kws]
        else:
            kws = ann['kwords']

        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)

        item['image'] = {}
        item['image']['data'] = image
        
        item['index'] = index

        ## VCs
        if self.use_vcs:
            if self.randkw_p is not None:
                num_kw = int(self.randkw_p * len(kws))
                vcs = random.choices(kws, k=num_kw)
            else:
                vcs = kws
            vcs = [' '.join(vcs)]
            vcs = self.tokenizer(vcs, padding='longest', truncation=True, max_length=55, return_tensors="pt") # tokenize kw with bert tokenizer
            item['image']['kwords_ids'] = vcs.input_ids[0]
            item['image']['kwords_masks'] = vcs.attention_mask[0]

            if self.aux_kwords:
                aux_words = [title]
                aux_kws = aux_words
                aux_kws = self.tokenizer(aux_kws, padding='longest', truncation=True, max_length=20, return_tensors="pt") # tokenize kw with bert tokenizer
                item['image']['aux_kwords_ids'] = aux_kws.input_ids[0]
                item['image']['aux_kwords_masks'] = aux_kws.attention_mask[0]

        ## recipe construction
        recipe = {}
        recipe['layer1'] = {}

        if self.bert:
            tokenised_caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=25, return_tensors="pt")['input_ids']
            recipe['layer1']['all'] = torch.LongTensor(tokenised_caption)
        else:
            kwords = [self.tokenizer(t, padding='max_length', truncation=True, max_length=7, return_tensors="pt")['input_ids'] for t in kws]
            kwords = torch.cat(kwords, dim=0)

            tokenised_caption = self.tokenizer(caption, padding='longest', truncation=True, max_length=25, return_tensors="pt")['input_ids']
            

            if self.only_captions:

                cap = torch.LongTensor(tokenised_caption)

                recipe['layer1']['title'] = cap.squeeze(0)
                recipe['layer1']['ingredients'] = cap
                recipe['layer1']['instructions'] = cap
            else:
                tokenised_title = self.tokenizer(title, padding='longest', truncation=True, max_length=15, return_tensors="pt")['input_ids'][0]

                recipe['layer1']['title'] = torch.LongTensor(tokenised_title)
                recipe['layer1']['ingredients'] = torch.LongTensor(kwords)
                recipe['layer1']['instructions'] = torch.LongTensor(tokenised_caption)


        item['recipe'] = recipe


        
        item['match'] = torch.FloatTensor([1])

        return item




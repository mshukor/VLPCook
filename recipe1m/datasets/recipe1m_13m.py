import os
import pickle
import torch
import torch.utils.data as data

from PIL import Image

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.datasets import utils
from bootstrap.datasets import transforms

from bootstrap.lib.options import Options
 
import json
import nltk  


import random 

random.seed(1234)
from random import choice



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


def get_token_ids(sentence, vocab):
    """
    get vocabulary tokens for each word in a sentence
    """
    tok_ids = []
    tokens = nltk.tokenize.word_tokenize(sentence.lower())

    tok_ids.append(vocab['<start>'])
    for token in tokens:
        if token in vocab:
            tok_ids.append(vocab[token])
        else:
            # unk words will be ignored
            tok_ids.append(vocab['<unk>'])
    tok_ids.append(vocab['<end>'])
    return tok_ids

def list2Tensors(input_list):
    """Given a list of lists of variable-length elements, return a 2D tensor padded with 0s
    """
    max_seq_len = max(map(len, input_list))
    output = [v + [0] * (max_seq_len - len(v)) for v in input_list]

    return torch.Tensor(output)


class Recipe1M_13M_debug(Dataset):
    def __init__(self, image_root, split, batch_size, nb_threads=4, transform=utils.default_image_tf(256, 224), 
                 max_ingrs=20,
                 max_instrs=20,
                 max_length_ingrs=15,
                 max_length_instrs=15,
                 path_ids=None, 
                 vocab_path=None, path_image_json=None, 
                 path_text_json=None, freq_mismatch=0, batch_sampler='random', 
                 ):

        super(Recipe1M_13M, self).__init__(image_root, split, batch_size, nb_threads)

        # Load ids 
        ids = json.load(open(path_ids,'r'))
        self.ids = ids[split]



        layer2 = json.load(open(path_image_json,'r'))
        self.layer2 = {}
        for d in layer2:
            self.layer2[d['id']] = d['images'] 

        layer1 = json.load(open(path_text_json,'r'))

        self.layer1 = {}
        for d in layer1:
            self.layer1[d['id']] = d

        self.image_root = image_root
        self.split = split
        self.transform = transform

        self.max_ingrs = max_ingrs
        self.max_instrs = max_instrs
        self.max_length_ingrs = max_length_ingrs
        self.max_length_instrs = max_length_instrs

        self.freq_mismatch = freq_mismatch
        self.batch_sampler = batch_sampler
        self.batch_size = batch_size
        self.nb_threads = nb_threads

        self.corrupted_img_ids = set()

    def __getitem__(self, idx):
        item = {}

        item_id = self.ids[idx]

        ## image
        is_match = True

        image_id = item_id
        item['match'] = torch.FloatTensor([1])


        image_entry = self.layer2[image_id]


        for img_name_ in image_entry:

            img_name = img_name_['id']

            img_name = '/'.join(img_name[:4])+'/'+img_name
            
            if self.transform is not None:
                try:
                    img = Image.open(os.path.join(self.image_root, img_name)).convert('RGB')
                    img = self.transform(img)
                    item['image'] = {}
                    item['image']['data'] = img
                    item['image']['path'] = img_name
                except:
                    self.corrupted_img_ids.add(img_name_['id'])


        return item


    def make_batch_loader(self, shuffle=True):
        if self.batch_sampler == 'random':
            if Options()['dataset'].get("debug", False):
                batch_loader = super(Recipe1M_13M, self).make_batch_loader(shuffle=False)
            else:
                batch_loader = super(Recipe1M_13M, self).make_batch_loader(shuffle=shuffle)
            Logger()('Dataset will be sampled with "random" batch_sampler.')
        else:
            raise ValueError()
        return batch_loader


    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return self.ids

    def get_vocab(self):
        try:
            return self.vocab_inv
        except:
            return None

class Recipe1M_13M(Dataset):
    def __init__(self, image_root, split, batch_size, nb_threads=4, transform=utils.default_image_tf(256, 224), 
                 max_ingrs=20,
                 max_instrs=20,
                 max_length_ingrs=15,
                 max_length_instrs=15,
                 path_ids=None, 
                 vocab_path=None, path_image_json=None, 
                 path_text_json=None, freq_mismatch=0, batch_sampler='random', tokenized_raw_text=False, 
                 use_vcs=False, 
                 kw_path=None, randkw_p=None, tokenizer=None, aux_kwords=False, aux_kw_path=None, randkw_p_aux=None, 
                 image_percentage=None, data_percentage=None
                 ):

        super(Recipe1M_13M, self).__init__(image_root, split, batch_size, nb_threads)

        # Load ids 
        ids = json.load(open(path_ids,'r'))
        self.ids = ids[split]

        self.data_percentage = data_percentage
        if self.data_percentage is not None and split == 'train':
            Logger()(self.data_percentage, len(self.ids))
            num_examples = int(float(self.data_percentage)*len(self.ids))
            self.ids = self.ids[:num_examples]
            Logger()('training on:', len(self.ids))

        #load vocabulary
        self.vocab = pickle.load(open(vocab_path, 'rb'))

        layer2 = json.load(open(path_image_json,'r'))
        self.layer2 = {}
        for d in layer2:
            self.layer2[d['id']] = d['images'] 


        self.tokenized_layer1 = tokenized_raw_text



        layer1 = json.load(open(path_text_json,'r'))

        if not self.tokenized_layer1:
            self.layer1 = {}
            for d in layer1:
                self.layer1[d['id']] = d
        else:
            self.layer1 = layer1

        self.image_root = image_root
        self.split = split
        self.transform = transform

        self.max_ingrs = max_ingrs
        self.max_instrs = max_instrs
        self.max_length_ingrs = max_length_ingrs
        self.max_length_instrs = max_length_instrs

        self.freq_mismatch = freq_mismatch
        self.batch_sampler = batch_sampler
        self.batch_size = batch_size
        self.nb_threads = nb_threads


        self.use_vcs = use_vcs
        self.aux_kwords = aux_kwords
        self.orig_ids_eval = Options()['dataset'].get("orig_ids_eval", False)
        self.image_percentage = image_percentage

        if self.use_vcs:
            Logger()('Load VCs...')
            self.image_path_to_kws = json.load(open(kw_path,'r'))
            # Logger()(list(self.image_path_to_kws.keys())[:30])
            if self.aux_kwords:
                self.image_path_to_aux_kws = json.load(open(aux_kw_path,'r'))


            if split == 'train':
                self.randkw_p = randkw_p
                self.randkw_p_aux = randkw_p_aux
            else:
                self.randkw_p = None
                self.randkw_p_aux = None
            Logger()('randkw_p...', self.randkw_p)

            self.dir_img_vcs = '/data/mshukor/data/recipe1m/recipe1M/images'
            self.tokenizer = tokenizer


    def __getitem__(self, idx):
        item = {}

        item_id = self.ids[idx]

        ## image
        if self.freq_mismatch > 0:
            is_match = torch.rand(1)[0] > self.freq_mismatch
        else:
            is_match = True

        if is_match:
            image_id = item_id
            item['match'] = torch.FloatTensor([1])
        else:
            n_index = int(torch.rand(1)[0] * len(self))
            image_id = n_index
            item['match'] = torch.FloatTensor([-1])

        image_entry = self.layer2[image_id]


        if self.split == 'train':
            if self.image_percentage is not None:
                num_images = max(1, int(float(self.image_percentage)*len(image_entry)))
                img_name = choice(image_entry[:num_images])
            else:
                img_name = choice(image_entry)
        else:
            img_name = image_entry[0]
        img_name = img_name['id']

        img_name = '/'.join(img_name[:4])+'/'+img_name
        img = Image.open(os.path.join(self.image_root, img_name)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        item['image'] = {}

        item['image']['data'] = img
        item['image']['index'] = image_id
        item['image']['path'] = img_name
        

        if self.use_vcs:
            if self.split == 'train' or self.orig_ids_eval:
                kw_path = item_id
            else:
                im_path = os.path.join(self.image_root,self.split, img_name)
                kw_path = im_path.replace(self.image_root, self.dir_img_vcs)

            if kw_path in self.image_path_to_kws:
                
                kwords = self.image_path_to_kws[kw_path]

                if self.randkw_p is not None:
                    num_kw = int(self.randkw_p * len(kwords))
                    kws = random.choices(kwords, k=num_kw)
                else:
                    kws = kwords

                if self.aux_kwords:
                    if kw_path in self.image_path_to_aux_kws:
                        aux_words = self.image_path_to_aux_kws[kw_path]

                        if self.randkw_p_aux is not None:
                            num_kw = int(self.randkw_p_aux * len(aux_words))
                            aux_kws = random.choices(aux_words, k=num_kw)
                        else:
                            aux_kws = aux_words
                    else:
                        aux_kws = ['food', 'food']

                    aux_kws = [' '.join(aux_kws)]
                    aux_kws = self.tokenizer(aux_kws, padding='longest', truncation=True, max_length=55, return_tensors="pt") # tokenize kw with bert tokenizer
                    item['image']['aux_kwords_ids'] = aux_kws.input_ids[0]
                    item['image']['aux_kwords_masks'] = aux_kws.attention_mask[0]

            else:
                kws = ['food', 'food']
                Logger()("kws not found")
                if self.aux_kwords:
                    aux_kws = ['food', 'food']
                    aux_kws = [' '.join(aux_kws)]
                    aux_kws = self.tokenizer(aux_kws, padding='longest', truncation=True, max_length=55, return_tensors="pt") # tokenize kw with bert tokenizer
                    item['image']['aux_kwords_ids'] = aux_kws.input_ids[0]
                    item['image']['aux_kwords_masks'] = aux_kws.attention_mask[0]



            kws = [' '.join(kws)]
            kws = self.tokenizer(kws, padding='longest', truncation=True, max_length=55, return_tensors="pt") # tokenize kw with bert tokenizer
            item['image']['kwords_ids'] = kws.input_ids[0]
            item['image']['kwords_masks'] = kws.attention_mask[0]



        # recipe
        recipe_entity = self.layer1[item_id]
        item['recipe'] = {}
        item['recipe']['layer1'] = {}

        if self.tokenized_layer1:

            item['recipe']['layer1']['title'] = torch.LongTensor(recipe_entity['title'])

            tokenized_ingrs = recipe_entity['ingredients'][:self.max_ingrs]
            tokenized_ingrs = [l[:self.max_length_ingrs] for l in tokenized_ingrs]
            max_len = max([len(l) for l in tokenized_ingrs])
            tokenized_ingrs = [l + (max_len - len(l))*[0] for l in tokenized_ingrs]
            item['recipe']['layer1']['ingredients'] = torch.LongTensor(tokenized_ingrs)

            tokenized_instrs = recipe_entity['instructions'][:self.max_instrs]
            tokenized_instrs = [l[:self.max_length_instrs] for l in tokenized_instrs]
            max_len = max([len(l) for l in tokenized_instrs])
            tokenized_instrs = [l + (max_len - len(l))*[0] for l in tokenized_instrs]
            item['recipe']['layer1']['instructions'] = torch.LongTensor(tokenized_instrs)

        else:
            

            title = recipe_entity['title']
            ingrs = recipe_entity['ingredients']
            instrs = recipe_entity['instructions']

            # turn text into indexes
            title = torch.Tensor(get_token_ids(title, self.vocab)[:self.max_length_instrs]).long()
            instrs = list2Tensors([get_token_ids(instr['text'], self.vocab)[:self.max_length_instrs] for instr in instrs[:self.max_instrs]]).long()
            ingrs = list2Tensors([get_token_ids(ingr['text'], self.vocab)[:self.max_length_ingrs] for ingr in ingrs[:self.max_ingrs]]).long()



            item['recipe']['layer1']['instructions'] = instrs
            item['recipe']['layer1']['ingredients'] = ingrs
            item['recipe']['layer1']['title'] = title

        item['recipe']['index'] = item_id

        return item


    def make_batch_loader(self, shuffle=True):
        if self.batch_sampler == 'random':
            if Options()['dataset'].get("debug", False):
                batch_loader = super(Recipe1M_13M, self).make_batch_loader(shuffle=False)
            else:
                batch_loader = super(Recipe1M_13M, self).make_batch_loader(shuffle=shuffle)
            Logger()('Dataset will be sampled with "random" batch_sampler.')
        else:
            raise ValueError()
        return batch_loader


    def __len__(self):
        return len(self.ids)

    def get_ids(self):
        return self.ids

    def get_vocab(self):
        try:
            return self.vocab_inv
        except:
            return None


def pad_input(input):
    """
    creates a padded tensor to fit the longest sequence in the batch
    """
    if len(input[0].size()) == 1:
        l = [len(elem) for elem in input]
        targets = torch.zeros(len(input), max(l)).long()
        for i, elem in enumerate(input):
            end = l[i]
            targets[i, :end] = elem[:end]
    else:
        n, l = [], []
        for elem in input:
            n.append(elem.size(0))
            l.append(elem.size(1))
        targets = torch.zeros(len(input), max(n), max(l)).long()
        for i, elem in enumerate(input):
            targets[i, :n[i], :l[i]] = elem
    return targets





if __name__ == '__main__':

    Logger(Options()['logs']['dir'])('lol')



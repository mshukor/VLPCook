import os
import lmdb
import pickle
import torch
import torch.utils.data as data

from PIL import Image

from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.datasets import utils
from bootstrap.datasets import transforms

from .batch_sampler import BatchSamplerTripletClassif
from bootstrap.lib.options import Options
 
import json



import random 


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

class DatasetLMDB(Dataset):

    def __init__(self, dir_data, split, batch_size, nb_threads):
        super(DatasetLMDB, self).__init__(dir_data, split, batch_size, nb_threads)
        self.dir_lmdb = os.path.join(self.dir_data, 'data_lmdb')

        self.path_envs = {}
        self.path_envs['ids'] = os.path.join(self.dir_lmdb, split, 'ids.lmdb')
        self.path_envs['numims'] = os.path.join(self.dir_lmdb, split, 'numims.lmdb')
        self.path_envs['impos'] = os.path.join(self.dir_lmdb, split, 'impos.lmdb')
        self.path_envs['imnames'] = os.path.join(self.dir_lmdb, split, 'imnames.lmdb')
        self.path_envs['ims'] = os.path.join(self.dir_lmdb, split, 'ims.lmdb')

        self.path_envs['classes'] = os.path.join(self.dir_lmdb, split, 'classes.lmdb')

        self.envs = {}
        self.envs['ids'] = lmdb.open(self.path_envs['ids'], readonly=True, lock=False)
        self.envs['classes'] = lmdb.open(self.path_envs['classes'], readonly=True, lock=False)

        self.txns = {}
        self.txns['ids'] = self.envs['ids'].begin(write=False, buffers=True)
        self.txns['classes'] = self.envs['classes'].begin(write=False, buffers=True)

        self.nb_recipes = self.envs['ids'].stat()['entries']


        self.path_pkl = os.path.join(self.dir_data, 'classes1M.pkl')
        #https://github.com/torralba-lab/im2recipe/blob/master/pyscripts/bigrams.py#L176
        with open(self.path_pkl, 'rb') as f:
            _ = pickle.load(f) # load the first line/object
            self.classes = pickle.load(f) # load the second line/object

        self.cname_to_cid = {v:k for k,v in self.classes.items()}

    def encode(self, value):
        return pickle.dumps(value)

    def decode(self, bytes_value):
        return pickle.loads(bytes_value)

    def get(self, index, env_name):
        buf = self.txns[env_name].get(self.encode(index))
        value = self.decode(bytes(buf))
        return value

    def _load_class(self, index):
        class_id = self.get(index, 'classes') - 1 # lua to python
        return torch.LongTensor([class_id]), self.classes[class_id]

    def __len__(self):
        return self.nb_recipes

    def true_nb_images(self):
        return self.envs['imnames'].stat()['entries']


class Images(DatasetLMDB):

    def __init__(self, dir_data, split, batch_size, nb_threads, image_from='database', 
        image_tf=utils.default_image_tf(256, 224), use_vcs=False, get_all_images=False, 
        kw_path=None, randkw_p=None, tokenizer=None, aux_kwords=False, aux_kw_path=None, randkw_p_aux=None, 
        random_kw=False, random_aux_kw=False):

        super(Images, self).__init__(dir_data, split, batch_size, nb_threads)
        self.image_tf = image_tf
        self.dir_img = os.path.join(dir_data,'recipe1M', 'images')

        self.envs['numims'] = lmdb.open(self.path_envs['numims'], readonly=True, lock=False)
        self.envs['impos'] = lmdb.open(self.path_envs['impos'], readonly=True, lock=False)
        self.envs['imnames'] = lmdb.open(self.path_envs['imnames'], readonly=True, lock=False)


        self.txns['numims'] = self.envs['numims'].begin(write=False, buffers=True)
        self.txns['impos'] = self.envs['impos'].begin(write=False, buffers=True)
        self.txns['imnames'] = self.envs['imnames'].begin(write=False, buffers=True)

        self.image_from = image_from
        if self.image_from == 'database':
            self.envs['ims'] = lmdb.open(self.path_envs['ims'], readonly=True, lock=False)
            self.txns['ims'] = self.envs['ims'].begin(write=False, buffers=True)
        self.use_vcs = use_vcs
        self.get_all_images = get_all_images
        self.aux_kwords = aux_kwords

        self.random_kw=random_kw
        self.random_aux_kw=random_aux_kw
 

        if self.use_vcs:
            Logger()('Load VCs...')
            self.image_path_to_kws = json.load(open(kw_path,'r'))

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



    def __getitem__(self, index):
        item = self.get_image(index)
        return item

    def format_path_img(self, raw_path):
        # "recipe1M/images/train/6/b/d/c/6bdca6e490.jpg"
        basename = os.path.basename(raw_path)
        path_img = os.path.join(self.dir_img,
                                self.split,
                                basename[0],
                                basename[1],
                                basename[2],
                                basename[3],
                                basename)
        return path_img

    def get_image(self, index):
        item = {}
        if self.get_all_images:
            item['samples'] = self._load_image_data(index)
        else:
            item['data'], item['index'], item['path'] = self._load_image_data(index)
        item['class_id'], item['class_name'] = self._load_class(index)
        if self.use_vcs:
            # print(self.image_path_to_kws.keys())
            kw_path = item['path'].replace(self.dir_img, self.dir_img_vcs)
            if self.random_kw:
                rand_index = random.choice(range(len(self)))
                _, _, rand_path = self._load_image_data(rand_index)
                kw_path = rand_path.replace(self.dir_img, self.dir_img_vcs)
            if kw_path in self.image_path_to_kws:
                
                kwords = self.image_path_to_kws[kw_path]

                if self.randkw_p is not None:
                    num_kw = int(self.randkw_p * len(kwords))
                    kws = random.choices(kwords, k=num_kw)
                else:
                    kws = kwords

                if self.aux_kwords:
                    if self.random_aux_kw:
                        rand_index = random.choice(range(len(self)))
                        _, _, rand_path = self._load_image_data(rand_index)
                        kw_path = rand_path.replace(self.dir_img, self.dir_img_vcs)
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
                    item['aux_kwords_ids'] = aux_kws.input_ids[0]
                    item['aux_kwords_masks'] = aux_kws.attention_mask[0]

            else:
                kws = ['food', 'food']
                Logger()("kws not found", item['path'])

            kws = [' '.join(kws)]
            kws = self.tokenizer(kws, padding='longest', truncation=True, max_length=55, return_tensors="pt") # tokenize kw with bert tokenizer
            item['kwords_ids'] = kws.input_ids[0]
            item['kwords_masks'] = kws.attention_mask[0]

        return item

    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _load_image_data(self, index):
        # select random image from list of images for that sample
        nb_images = self.get(index, 'numims')
        if self.get_all_images:
            images = []
            for im_idx in range(nb_images):

                index_img = self.get(index, 'impos')[im_idx] - 1 # lua to python

                path_img = self.format_path_img(self.get(index_img, 'imnames'))

                if self.image_from == 'pil_loader':
                    image_data = self._pil_loader(path_img) 
                elif self.image_from == 'database':
                    image_data = self.get(index_img, 'ims')

                if self.image_tf is not None:
                    image_data = self.image_tf(image_data)
                item = {}
                item['data'], item['index'], item['path'] = image_data, index_img, path_img
                images.append(item)
            return images
        else:
            try:
                if Options()['dataset'].get("debug", False):
                    im_idx = 0
                else:
                    im_idx = torch.randperm(nb_images)[0]
            except:
                im_idx = torch.randperm(nb_images)[0]

            index_img = self.get(index, 'impos')[im_idx] - 1 # lua to python

            path_img = self.format_path_img(self.get(index_img, 'imnames'))

            if self.image_from == 'pil_loader':
                image_data = self._pil_loader(path_img) 
            elif self.image_from == 'database':
                image_data = self.get(index_img, 'ims')

            if self.image_tf is not None:
                image_data = self.image_tf(image_data)
                
            return image_data, index_img, path_img


class Recipes_raw(DatasetLMDB):

    def __init__(self, dir_data, split, batch_size, nb_threads):
        super(Recipes_raw, self).__init__(dir_data, split, batch_size, nb_threads)

        # ~added for visu
        import json
        self.path_layer1 = os.path.join(dir_data, 'text', 'tokenized_layer1.json')
        with open(self.path_layer1, 'r') as f:
            self.layer1 = json.load(f)
        self.envs['ids'] = lmdb.open(self.path_envs['ids'], readonly=True, lock=False)
        # # ~end

        self.with_titles = Options()['model']['network'].get('with_titles', False)
        self.tokenized_raw_text = Options()['dataset'].get('tokenized_raw_text', False)
        self.max_instrs_len = Options()['dataset'].get('max_instrs_len', 20)
        self.max_ingrs_len = Options()['dataset'].get('max_ingrs_len', 15)
        self.max_instrs = Options()['dataset'].get('max_instrs', 20)
        self.max_ingrs = Options()['dataset'].get('max_ingrs', 20)

        self.remove_list = Options()['dataset'].get('remove_list', None)
        self.interchange_ingrd_instr = Options()['dataset'].get('interchange_ingrd_instr', None)
        Logger()('recipe elements to remove:', self.remove_list)


    def __getitem__(self, index):
        item = self.get_recipe(index)
        return item

    def get_recipe(self, index):
        item = {}
        item['class_id'], item['class_name'] = self._load_class(index)
 
        item['index'] = index
        # ~added for visu
        item['ids'] = self.get(index, 'ids')
        item['layer1'] = self.layer1[item['ids']]

        item['layer1']['title'] = torch.LongTensor(item['layer1']['title'])
        if self.remove_list is not None and 'title' in self.remove_list:
            item['layer1']['title'] = torch.LongTensor([167838,  178987, 59198]) # [start ukn end] tokens

        if self.remove_list is not None and 'ingredients' in self.remove_list:
            item['layer1']['ingredients'] = torch.LongTensor([[167838,  178987, 59198]])
        else:
            tokenized_ingrs = item['layer1']['ingredients'][:self.max_ingrs]
            tokenized_ingrs = [l[:self.max_ingrs_len] for l in tokenized_ingrs]
            max_len = max([len(l) for l in tokenized_ingrs])
            tokenized_ingrs = [l + (max_len - len(l))*[0] for l in tokenized_ingrs]
            item['layer1']['ingredients'] = torch.LongTensor(tokenized_ingrs)

        if self.remove_list is not None and 'instructions' in self.remove_list:
            item['layer1']['instructions'] = torch.LongTensor([[167838,  178987, 59198]])
        else:
            tokenized_instrs = item['layer1']['instructions'][:self.max_instrs]
            tokenized_instrs = [l[:self.max_instrs_len] for l in tokenized_instrs]
            max_len = max([len(l) for l in tokenized_instrs])
            tokenized_instrs = [l + (max_len - len(l))*[0] for l in tokenized_instrs]
            item['layer1']['instructions'] = torch.LongTensor(tokenized_instrs)

        if self.interchange_ingrd_instr is not None:
            tmp = item['layer1']['instructions'].clone()
            item['layer1']['instructions'] = item['layer1']['ingredients'].clone()
            item['layer1']['ingredients'] = tmp

        # ~end
        return item



class Recipe1M(DatasetLMDB):

    def __init__(self, dir_data, split, batch_size=100, nb_threads=4, freq_mismatch=0.,
            batch_sampler='triplet_classif',
            image_from='database', image_tf=utils.default_image_tf(256, 224), 
            use_vcs=False, kw_path=None, randkw_p=None, tokenizer=None,
            aux_kwords=False, aux_kw_path=None, randkw_p_aux=None, 
            random_kw=False, random_aux_kw=False):
        super(Recipe1M, self).__init__(dir_data, split, batch_size, nb_threads)


        self.images_dataset = Images(dir_data, split, batch_size, nb_threads, image_from=image_from, 
            image_tf=image_tf, use_vcs=use_vcs, kw_path=kw_path, randkw_p=randkw_p, tokenizer=tokenizer, 
            aux_kwords=aux_kwords, aux_kw_path=aux_kw_path, randkw_p_aux=randkw_p_aux, 
            random_kw=random_kw, random_aux_kw=random_aux_kw)
        self.tokenized_raw_text = Options()['dataset'].get('tokenized_raw_text', False)
        self.dataset_revamping = Options()['dataset'].get('dataset_revamping', False)
        if self.tokenized_raw_text:
            self.recipes_dataset = Recipes_raw(dir_data, split, batch_size, nb_threads)
        else:
            raise NotImplementedError("Only raw text is supported")
        self.freq_mismatch = freq_mismatch
        self.batch_sampler = batch_sampler

        if self.split == 'train' and self.batch_sampler == 'triplet_classif':
            self.indices_by_class = self._make_indices_by_class()


 
    def _make_indices_by_class(self):
        Logger()('Calculate indices by class...')
        indices_by_class = [[] for class_id in range(len(self.classes))]
        for index in range(len(self.recipes_dataset)):
            class_id = self._load_class(index)[0][0] # bcause (class_id, class_name) and class_id is a Tensor
            indices_by_class[class_id].append(index)
        Logger()('Done!')
        return indices_by_class

    def make_batch_loader(self, shuffle=True):
        if self.split in ['val', 'test'] or self.batch_sampler == 'random':
            if Options()['dataset'].get("debug", False):
                batch_loader = super(Recipe1M, self).make_batch_loader(shuffle=False)
            else:
                batch_loader = super(Recipe1M, self).make_batch_loader(shuffle=shuffle)
            Logger()('Dataset will be sampled with "random" batch_sampler.')
        elif self.batch_sampler == 'triplet_classif':
            batch_sampler = BatchSamplerTripletClassif(
                self.indices_by_class,
                self.batch_size,
                pc_noclassif=0.5,
                nb_indices_same_class=2)
            batch_loader = data.DataLoader(self,
                num_workers=self.nb_threads,
                batch_sampler=batch_sampler,
                pin_memory=True,
                collate_fn=self.items_tf())
            Logger()('Dataset will be sampled with "triplet_classif" batch_sampler.')
        else:
            raise ValueError()
        return batch_loader

    def __getitem__(self, index):
        #ids = self.data['ids'][index]
        item = {}
        item['index'] = index
        item['recipe'] = self.recipes_dataset[index]

        if self.freq_mismatch > 0:
            is_match = torch.rand(1)[0] > self.freq_mismatch
        else:
            is_match = True

        if is_match:
            item['image'] = self.images_dataset[index]
            item['match'] = torch.FloatTensor([1])
        else:
            n_index = int(torch.rand(1)[0] * len(self))
            item['image'] = self.images_dataset[n_index]
            item['match'] = torch.FloatTensor([-1])
        return item





if __name__ == '__main__':

    Logger(Options()['logs']['dir'])('lol')



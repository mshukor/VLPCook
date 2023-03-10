from bootstrap.lib.options import Options
from .recipe1m import Recipe1M 
from .vlp import RecipeVLP
from .tokenization_bert import BertTokenizer

from torchvision import transforms

from .randaugment import RandomAugment
from PIL import Image
import os 

from .recipe1m_13m import Recipe1M_13M


def factory(engine=None):
    dataset = {}

    if Options()['dataset']['name'] == 'recipe1m':    
        
        if Options()['dataset'].get('train_split', None):
            dataset['train'] = factory_recipe1m(Options()['dataset']['train_split'])

        if Options()['dataset'].get('eval_split', None): 
            dataset['eval'] = factory_recipe1m(Options()['dataset']['eval_split'])
            
    elif Options()['dataset']['name'] == 'recipevlp':  
        dataset['train'] = factory_recipevlp()

    elif Options()['dataset']['name'] == 'recipe1m_13m':
        if Options()['dataset'].get('train_split', None):
            dataset['train'] = factory_recipe1m_13m(Options()['dataset']['train_split'])
        if Options()['dataset'].get('eval_split', None): 
            dataset['eval'] = factory_recipe1m_13m(Options()['dataset']['eval_split'])

    else:
        raise ValueError()

    return dataset

def factory_recipe1m_13m(split):

    use_vcs = Options()['dataset'].get('use_vcs', False)

    kw_path = Options()['dataset'].get('kw_path', False)
    randkw_p = Options()['dataset'].get('randkw_p', None)

    randkw_p_aux = Options()['dataset'].get('randkw_p_aux', None)
    aux_kw_path = Options()['dataset'].get('aux_kw_path', False)
    aux_kwords = Options()['model.network'].get('aux_kwords', False)
    
    image_percentage = Options()['dataset'].get('image_percentage', None)

    data_percentage = Options()['dataset'].get('data_percentage', None)


    if use_vcs:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    else:
        tokenizer = None

    if kw_path:
        kw_path_split = Options()['dataset.kw_path'].get(split, None)
    else:
        kw_path_split = None

    if aux_kw_path:
        aux_kw_path_split = Options()['dataset.aux_kw_path'].get(split, None)
    else:
        aux_kw_path_split = None


    dataset = Recipe1M_13M(
        Options()['dataset']['dir'],
        split=split,
        batch_size=Options()['dataset']['batch_size'],
        nb_threads=Options()['dataset']['nb_threads'],
        freq_mismatch=Options()['dataset']['freq_mismatch'],
        batch_sampler=Options()['dataset']['batch_sampler'],
        path_ids=Options()['dataset']['path_ids'],
        vocab_path=Options()['dataset']['vocab_path'], 
        path_image_json=Options()['dataset']['path_image_json'], 
        path_text_json=Options()['dataset']['path_text_json'],
        tokenized_raw_text=Options()['dataset']['tokenized_raw_text'],
        use_vcs=use_vcs, kw_path=kw_path_split, 
        randkw_p=randkw_p, tokenizer=tokenizer,
        aux_kwords=aux_kwords, aux_kw_path=aux_kw_path_split, randkw_p_aux=randkw_p_aux, 
        image_percentage=image_percentage, data_percentage=data_percentage)


    return dataset

def factory_recipe1m(split):
    use_vcs = Options()['dataset'].get('use_vcs', False)

    kw_path = Options()['dataset'].get('kw_path', False)
    randkw_p = Options()['dataset'].get('randkw_p', None)

    randkw_p_aux = Options()['dataset'].get('randkw_p_aux', None)
    aux_kw_path = Options()['dataset'].get('aux_kw_path', False)
    aux_kwords = Options()['model.network'].get('aux_kwords', False)
    
    random_kw = Options()['dataset'].get('random_kw', False)
    random_aux_kw = Options()['dataset'].get('random_aux_kw', False)


    if use_vcs:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    else:
        tokenizer = None

    if kw_path:
        kw_path_split = Options()['dataset.kw_path'].get(split, None)
    else:
        kw_path_split = None

    if aux_kw_path:
        aux_kw_path_split = Options()['dataset.aux_kw_path'].get(split, None)
    else:
        aux_kw_path_split = None

    dataset = Recipe1M(
        Options()['dataset']['dir'],
        split,
        batch_size=Options()['dataset']['batch_size'],
        nb_threads=Options()['dataset']['nb_threads'],
        freq_mismatch=Options()['dataset']['freq_mismatch'],
        batch_sampler=Options()['dataset']['batch_sampler'],
        image_from=Options()['dataset']['image_from'],
        use_vcs=use_vcs, kw_path=kw_path_split, 
        randkw_p=randkw_p, tokenizer=tokenizer,
        aux_kwords=aux_kwords, aux_kw_path=aux_kw_path_split, randkw_p_aux=randkw_p_aux, 
        random_kw=random_kw, random_aux_kw=random_aux_kw)
    return dataset


def factory_recipevlp():
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(Options()['dataset']['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    

    use_vcs = Options()['dataset'].get('use_vcs', False)

    randkw_p = Options()['dataset'].get('randkw_p', None)

    aux_kwords = Options()['model.network'].get('aux_kwords', None)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)

    #### Dataset #### 
    train_files = []
    for p in Options()['dataset']['train_file']:
        train_files.append(os.path.join(Options()['dataset']['data_json_dir'], p))

    use_tags = Options()['dataset'].get('use_tags', False)
    only_captions = Options()['dataset'].get('only_captions', False)

    bert = 'bert' in Options()['model.network'].get('recipe_encoder', False) and Options()['dataset'].get('bert', False)


    dataset = RecipeVLP(
        ann_file=train_files,
        transform=pretrain_transform,
        data_dir=Options()['dataset']['dir'],
        batch_size=Options()['dataset']['batch_size'],
        nb_threads=Options()['dataset']['nb_threads'],
        tokenizer=tokenizer, use_tags=use_tags, 
        only_captions=only_captions, bert=bert, 
        use_vcs=use_vcs, randkw_p=randkw_p, aux_kwords=aux_kwords)

    return dataset
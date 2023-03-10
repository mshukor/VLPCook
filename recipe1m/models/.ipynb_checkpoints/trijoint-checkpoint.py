import torch
import torch.nn as nn

from bootstrap.lib.options import Options
from bootstrap.datasets import transforms
from bootstrap.models.model import Model

from . import networks
from . import criterions
from . import metrics

from bootstrap.lib.logger import Logger
import os 

from .networks.tokenization_bert import BertTokenizer
import pickle 

from .networks.recipe_networks.xbert import BertEmbeddings


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint

class Trijoint(Model):

    def __init__(self,
                 opt,
                 nb_classes,
                 modes=['train', 'eval'],
                 engine=None,
                 cuda_tf=transforms.ToCuda):
        super(Trijoint, self).__init__(engine, cuda_tf=cuda_tf)
        if Options()['misc'].get("device_id", False):
            self.device_id = Options()['misc.device_id']

        if Options()['misc']['cuda']:
            if Options()['misc'].get("device_id", False):
                ids = Options()['misc.device_id']
                if isinstance(ids, list):
                    self.device = torch.device('cuda:'+str(ids[0]))
                else:
                    self.device = torch.device('cuda:'+str(ids))    
            else:
                self.device = torch.device('cuda')

        self.cross_encoder_model = Options()['model.network'].get('cross_encoder', False)

        if 'albef' in opt['network'].recipe_encoder:
            tokenizer = BertTokenizer.from_pretrained(opt['network.text_encoder'], local_files_only=True)
            model = networks.ALBEF(text_encoder=opt['network.text_encoder'],
                tokenizer=tokenizer,
                config=opt['network'])

            if opt['network'].get('checkpoint', False):
                checkpoint_path = opt['network.checkpoint']
                checkpoint = torch.load(checkpoint_path, map_location='cpu') 
                state_dict = checkpoint['model']
                
                # reshape positional embedding to accomodate for image resolution change
                pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
                state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
                if model.use_teacher:
                    m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                    state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
                
                for key in list(state_dict.keys()):
                    if 'bert' in key:
                        encoder_key = key.replace('bert.','')         
                        state_dict[encoder_key] = state_dict[key] 
                        del state_dict[key]                
                msg = model.load_state_dict(state_dict,strict=False)  
                
                print('load checkpoint from %s'%checkpoint_path)
                print(msg)  

            if opt['network'].get('vocab_size', False):
                vocab_size = opt['network.vocab_size']
            else:
                path_vocab = opt['network'].get('path_vocab', None)
                with open(path_vocab,'rb') as f:
                    data = pickle.load(f)
                vocab_size = len(data)
            print("Vocab size:", vocab_size)
            if vocab_size != model.bert_config.vocab_size:
                model.bert_config.vocab_size = vocab_size
                model.text_encoder.embeddings = BertEmbeddings(model.bert_config)

            self.network = model.to(self.device)
            # print(self.network)
        elif 'tfoodvicha' in opt['network'].recipe_encoder:
            self.network = networks.TfoodViCHA(
                opt['network'],
                nb_classes,
                with_classif=opt['with_classif'])
        elif 'vicha' in opt['network'].recipe_encoder:
            tokenizer = BertTokenizer.from_pretrained(opt['network.text_encoder'], local_files_only=True)
            model = networks.kw_img_ViCHA(text_encoder=opt['network.text_encoder'],
                tokenizer=tokenizer,
                config=opt['network'])

            if opt['network'].get('checkpoint', False):
                checkpoint_path = opt['network.checkpoint']
                checkpoint = torch.load(checkpoint_path, map_location='cpu') 
                state_dict = checkpoint['model']
                
                # reshape positional embedding to accomodate for image resolution change
                pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
                state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
                if model.use_teacher:
                    m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                    state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
                
                for key in list(state_dict.keys()):
                    if 'bert' in key:
                        encoder_key = key.replace('bert.','')         
                        state_dict[encoder_key] = state_dict[key] 
                        del state_dict[key]                
                msg = model.load_state_dict(state_dict,strict=False)  
                
                print('load checkpoint from %s'%checkpoint_path)
                print(msg)  

            if opt['network'].get('vocab_size', False):
                vocab_size = opt['network.vocab_size']
            else:
                path_vocab = opt['network'].get('path_vocab', None)
                with open(path_vocab,'rb') as f:
                    data = pickle.load(f)
                vocab_size = len(data)
            print("Vocab size:", vocab_size)
            if vocab_size != model.bert_config.vocab_size:
                model.bert_config.vocab_size = vocab_size
                model.text_encoder.embeddings = BertEmbeddings(model.bert_config)

            self.network = model.to(self.device)

        else:
            self.network = networks.CrossTrijoint(
                opt['network'],
                nb_classes,
                with_classif=opt['with_classif'], opt_all=Options())


        self.criterions = {}
        self.metrics = {}


        self.itm_loss_weight = Options()['model.criterion'].get('itm_loss_weight', 0)
        if self.itm_loss_weight > 0:
            self.trijoint_metric = Options()['model.metric'].get('trijoint', False) 
        else: 
            self.trijoint_metric = True
        print(modes)
        if 'pretrain' in modes:
            self.criterions['train'] = criterions.Trijoint(
                opt['criterion'],
                nb_classes,
                opt['network']['dim_emb'],
                with_classif=opt['with_classif'],
                engine=engine)
        elif 'train' in modes:
            self.criterions['train'] = criterions.Trijoint(
                opt['criterion'],
                nb_classes,
                opt['network']['dim_emb'],
                with_classif=opt['with_classif'],
                engine=engine)

            if self.cross_encoder_model and not self.trijoint_metric:
                self.metrics['train'] = metrics.CrossTrijoint(
                opt['metric'],
                with_classif=opt['with_classif'],
                engine=engine,
                mode='train')
            else:
                self.metrics['train'] = metrics.Trijoint(
                    opt['metric'],
                    with_classif=opt['with_classif'],
                    engine=engine,
                    mode='train')

        if 'eval' in modes:
            if self.cross_encoder_model and not self.trijoint_metric:
                if any([b in opt['network'].recipe_encoder for b in ['albef', 'vicha']]):
                    self.metrics['eval'] = metrics.CrossALBEF(
                        opt['metric'],
                        with_classif=opt['with_classif'],
                        engine=engine,
                        mode='eval')
                else:
                    self.metrics['eval'] = metrics.CrossTrijoint(
                        opt['metric'],
                        with_classif=opt['with_classif'],
                        engine=engine,
                        mode='eval')
            else:
                self.metrics['eval'] = metrics.Trijoint(
                    opt['metric'],
                    with_classif=opt['with_classif'],
                    engine=engine,
                    mode='eval')




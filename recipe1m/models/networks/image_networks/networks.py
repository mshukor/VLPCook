import pretrainedmodels
import torch
import torch.nn as nn
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

import clip
import timm 

 

class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet')
        self.dim_out = self.resnet.last_linear.in_features
        self.resnet.last_linear = None

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ViT_CLIP_custom(nn.Module):
    def __init__(self, model_name='ViT-B/16', device=None, cat_pos=0, cat_pos_aux=0):
        super(ViT_CLIP_custom, self).__init__()
        self.device = device
        clip_model, _ = clip.load("ViT-B/16", device=self.device)

        self.clip_model = clip_model.visual
        self.cat_pos = cat_pos
        self.cat_pos_aux = cat_pos_aux
        if self.cat_pos > 0 or self.cat_pos_aux > 0:
            self.clip_model.transformer.resblocks = nn.ModuleList([*self.clip_model.transformer.resblocks])

    def forward(self, x, external_features=None, aux_external_features=None):
        x = self.clip_model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.positional_embedding.to(x.dtype)

        if aux_external_features is not None:
            aux_external_features = aux_external_features.type(x.dtype)

        if self.cat_pos == 0:
            if external_features is not None:
                x = torch.cat((x, external_features), dim=1)

        if self.cat_pos_aux == 0:
            if aux_external_features is not None:
                x = torch.cat((x, aux_external_features), dim=1)


        x = self.clip_model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        if external_features is not None:
            external_features = external_features.permute(1, 0, 2)  # NLD -> LND
        if aux_external_features is not None:
            aux_external_features = aux_external_features.permute(1, 0, 2)  # NLD -> LND

        if self.cat_pos > 0 or self.cat_pos_aux > 0:
            for i,blk in enumerate(self.clip_model.transformer.resblocks):
                if i == self.cat_pos and self.cat_pos > 0:
                    x = torch.cat((x, external_features), dim=0)

                if i == self.cat_pos_aux and self.cat_pos_aux > 0:
                    x = torch.cat((x, external_features), dim=0)
                x = blk(x) 
        else:
            x = self.clip_model.transformer(x)

        
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip_model.ln_post(x)
        return x



class ViT_timm_custom(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', unpooled=False, 
        all_tokens=False, cat_pos=0, cat_pos_aux=0):
        super(ViT_timm_custom, self).__init__()

        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.dim_out = self.encoder.head.in_features
        self.unpooled = unpooled
        self.all_tokens = all_tokens
        self.cat_pos = cat_pos
        self.cat_pos_aux = cat_pos_aux
        if self.cat_pos > 0:
            self.encoder.blocks = nn.ModuleList([*self.encoder.blocks])

    def forward_features(self, x, external_features=None, aux_external_features=None):
        """https://github.com/rwightman/pytorch-image-models/issues/657"""
        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.encoder.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.encoder.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)

        if aux_external_features is not None:
            aux_external_features = aux_external_features.type(x.dtype)
        if self.cat_pos == 0:
            if external_features is not None:
                x = torch.cat((x, external_features), dim=1)

        if self.cat_pos_aux == 0:
            if aux_external_features is not None:
                x = torch.cat((x, aux_external_features), dim=1)

        if self.cat_pos > 0 or self.cat_pos_aux > 0:
            for i,blk in enumerate(self.encoder.blocks):
                if i == self.cat_pos and self.cat_pos > 0:
                    x = torch.cat((x, external_features), dim=1)

                if i == self.cat_pos_aux and self.cat_pos_aux > 0:
                    x = torch.cat((x, external_features), dim=1)
                x = blk(x) 
        else:
            x = self.encoder.blocks(x)
        if self.unpooled:
            if self.all_tokens:
                return x 
            if self.encoder.dist_token is None:
                return x[:, 1:]
            else:
                return x[:, 2:]
        else:
            x = self.encoder.norm(x)
            if self.encoder.dist_token is None:
                return self.encoder.pre_logits(x[:, 0])
            else:
                return x[:, 0], x[:, 1]

    def forward(self, x, external_features=None, aux_external_features=None):
        x = self.forward_features(x, external_features=external_features, aux_external_features=aux_external_features)
        return x



class ViT(nn.Module):

    def __init__(self, model_name='vit_base_patch16_224'):
        super(ViT, self).__init__()
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.dim_out = self.encoder.head.in_features
    

    def forward(self, x):
        x = self.encoder.forward_features(x)
        return x


class ImageEmbedding(nn.Module):

    def __init__(self, opt, device=None):
        super(ImageEmbedding, self).__init__()
        self.dim_emb = opt['dim_emb']
        self.activations = opt.get('activations', None)
        self.device = device
        self.get_tokens = Options()['model.criterion.retrieval_strategy'].get('get_tokens', False)
        self.all_tokens = opt.get('vit_all_tokens', False)
        self.cross_encoder = opt.get('cross_encoder', False)

        self.cat_pos = opt.get('cat_pos', 0)
        self.cat_pos_aux = opt.get('cat_pos_aux', 0)
        # modules
        if opt.get('image_backbone_name', False):
            if 'vit' in opt['image_backbone_name']:
                Logger()('Loading ViT model...', self.get_tokens)
                if not self.get_tokens:
                    self.convnet = ViT(model_name=opt['image_backbone_name'])
                else:
                    self.convnet = ViT_timm_custom(model_name=opt['image_backbone_name'], 
                        unpooled=self.get_tokens, all_tokens=self.all_tokens, 
                        cat_pos=self.cat_pos, cat_pos_aux=self.cat_pos_aux)
                dim_out = self.convnet.dim_out
                self.dtype = self.convnet.encoder.patch_embed.proj.weight.dtype

            elif 'clip' in opt['image_backbone_name']:
                Logger()('Loading CLIP Visual model...')
                if not self.all_tokens:
                    model, _ = clip.load("ViT-B/16", device=self.device) # jit=False
                    model.float()
                    self.convnet = model.visual
                    dim_out = self.convnet.output_dim
                    self.dtype = self.convnet.conv1.weight.dtype
                else:
                    self.convnet = ViT_CLIP_custom(device=self.device)
                    self.convnet.clip_model.float()
                    dim_out = self.convnet.clip_model.proj.size()[0]
                    self.dtype = self.convnet.clip_model.conv1.weight.dtype
            else:
                raise NotImplementedError
        else: 
            self.convnet = ResNet()
            dim_out = self.convnet.dim_out
            self.dtype = self.convnet.resnet.conv1.weight.dtype
        self.dim_out = dim_out
        self.fc = nn.Linear(dim_out, self.dim_emb)

    def forward(self, image, external_features=None, aux_external_features=None):
        if external_features is not None:
            x_ = self.convnet(image['data'].type(self.dtype), external_features.type(self.dtype), 
                aux_external_features=aux_external_features)
        else:
            x_ = self.convnet(image['data'].type(self.dtype))
        x = x_
        if not self.cross_encoder:
            x = self.fc(x)
        if self.activations is not None and not self.cross_encoder:
            for name in self.activations:                
                x = nn.functional.__dict__[name](x)
        return x 
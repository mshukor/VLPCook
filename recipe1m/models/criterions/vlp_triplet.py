import torch
import numpy as np
import torch.nn as nn
import scipy.linalg as la
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options
import os 
import torch.nn.functional as F
 
class VLPTriplet(nn.Module):

    def __init__(self, opt, dim_emb, engine=None):
        self.alpha = opt['retrieval_strategy']['margin']
        self.sampling = opt['retrieval_strategy']['sampling']
        self.nb_samples = opt['retrieval_strategy'].get('nb_samples', 1)
        self.substrategy = opt['retrieval_strategy'].get('substrategy', ['IRR'])
        self.aggregation = opt['retrieval_strategy'].get('aggregation', 'mean')
        self.id_background = opt['retrieval_strategy'].get('id_background', 0)
        self.dim_emb = dim_emb
        
        if Options()['misc']['cuda']:
            if Options()['misc'].get("device_id", False):
                ids = Options()['misc.device_id']
                if isinstance(ids, list):
                    self.device = torch.device('cuda:'+str(ids[0]))
                else:
                    self.device = torch.device('cuda:'+str(ids))
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if 'substrategy_weights' in opt['retrieval_strategy']:
            self.substrategy_weights = opt['retrieval_strategy']['substrategy_weights']
            if len(self.substrategy) > len(self.substrategy_weights):
                Logger()('Incorrect number of items in substrategy_weights (expected {}, got {})'.format(
                    len(self.substrategy), len(self.substrategy_weights)), Logger.ERROR)
            elif len(self.substrategy) < len(self.substrategy_weights):
                Logger()('Higher number of items in substrategy_weights than expected ({}, got {}). Discarding exceeding values'.format(
                    len(self.substrategy), len(self.substrategy_weights)), Logger.WARNING)
        else:
            Logger()('No substrategy_weights provided, automatically setting all items to 1', Logger.WARNING)
            self.substrategy_weights = [1.0] * len(self.substrategy)

        if opt['retrieval_strategy'].get('margin_params', False):
            self.increment_margin = opt['retrieval_strategy.margin_params'].get('increment_margin', False)
            self.adaptive_margin = opt['retrieval_strategy.margin_params'].get('adaptive_margin', False)

            if self.increment_margin:
                self.increment = opt['retrieval_strategy.margin_params'].get('increment', 0.005)
                self.max_margin = opt['retrieval_strategy.margin_params'].get('max_margin', 0.3)
                engine.register_hook('train_on_end_epoch', self.increment_alpha)
                print('increment alpha')
                resume = Options()['exp'].get('resume', False)
                if resume:
                    map_location = None if Options().get('misc.cuda', False) else 'cpu'
                    path_template = os.path.join(Options()['exp']['dir'], 'ckpt_{}_{}.pth.tar')
                    engine_state = torch.load(path_template.format(Options()['exp']['resume'], 'engine'), map_location=map_location)
                    print('engine.epoch', engine_state['epoch'])
                    epoch = engine_state['epoch']
                    print('epoch:', epoch)
                    self.alpha += self.increment*epoch
                    if self.alpha > self.max_margin:
                        self.alpha = self.max_margin
            elif self.adaptive_margin:
                self.min_margin = torch.tensor(opt['retrieval_strategy.margin_params'].get('min_margin', 0.05))
                self.max_margin = torch.tensor(opt['retrieval_strategy.margin_params'].get('max_margin', 0.7))
                engine.register_hook('train_on_end_epoch', self.print_margin)
        else:
            self.adaptive_margin = False


    def print_margin(self, ):
        Logger()('margin irr = ', self.alpha_irr)
        Logger()('margin rii = ', self.alpha_rii)

    def increment_alpha(self, ):
        if self.alpha < self.max_margin:
            self.alpha += self.increment 
        Logger()('margin = ', self.alpha)

    def calculate_cost(self, cost, enable_naive=True):
        if self.sampling == 'max_negative':
            ans,_ = torch.sort(cost, dim=1, descending=True)
        elif self.sampling == 'semi_hard':
            noalpha = cost - self.alpha
            mask = (noalpha <= 0)
            noalpha.masked_scatter_(mask, noalpha.max().expand_as(mask))
            ans, __argmax = torch.sort(noalpha, dim=1, descending=True)
            ans += self.alpha
        elif self.sampling == 'prob_negative':
            indexes = torch.multinomial(cost, cost.size(1))
            ans = torch.gather(cost, 1, indexes.detach())
        elif self.sampling == 'random':
            if enable_naive:
                Logger()('Random triplet strategy is outdated and does not work with non-square matrices :(', Logger.ERROR)
                indexes = la.hankel(np.roll(np.arange(cost.size(0)),-1), np.arange(cost.size(1))) # anti-circular matrix
                indexes = cost.data.new(indexes.tolist()).long()
                ans = torch.gather(cost, 1, indexes.detach())
            else:
                Logger()('Random triplet strategy not allowed with this configuration', Logger.ERROR)
        else:
            Logger()('Unknown substrategy {}.'.format(self.sampling), Logger.ERROR)
            
        return ans[:,:self.nb_samples]

    def add_cost(self, name, cost, bad_pairs, losses):
        invalid_pairs = (cost == 0).float().sum()
        bad_pairs['bad_pairs_{}'.format(name)] = invalid_pairs / cost.numel()
        if self.aggregation == 'mean':
            losses['loss_{}'.format(name)] = cost.mean() * self.substrategy_weights[self.substrategy.index(name)]
        elif self.aggregation == 'valid':
            valid_pairs = cost.numel() - invalid_pairs
            if cost.sum() == 0:
                losses['loss_{}'.format(name)] = cost.sum()
            else:
                losses['loss_{}'.format(name)] = cost.sum() * self.substrategy_weights[self.substrategy.index(name)] / valid_pairs
        else:
            Logger()('Unknown aggregation strategy {}.'.format(self.aggregation), Logger.ERROR)



    def __call__(self, input1, input2, target):
        bad_pairs = {}
        losses = {}


        # Prepare instance samples (matched pairs)
        matches = target.squeeze(1) == 1 # To support -1 or 0 as mismatch
        instance_input1 = input1[matches].view(matches.sum().int().item(), input1.size(1))

        instance_input2 = input2[matches].view(matches.sum().int().item(), input2.size(1))

          
        # Instance-based triplets
        if len(set(['IRR', 'RII', 'IRI', 'RIR', 'LIFT']).intersection(self.substrategy)) > 0:
            distances = self.dist(instance_input1, instance_input2)
            if 'IRR' in self.substrategy:
                try:
                    cost = distances.diag().unsqueeze(1) - distances + self.alpha.to(distances.device) # all triplets
                except AttributeError:
                    cost = distances.diag().unsqueeze(1) - distances + self.alpha

                cost[cost < 0] = 0 # hinge
                cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                if self.adaptive_margin:
                    self.alpha_irr = min(max(self.min_margin, self.max_margin*(1 -  ((cost.numel() - (cost == 0).float().sum()) / cost.numel()).item())), self.max_margin)
                    cost = distances.diag().unsqueeze(1) - distances + self.alpha_irr
                    cost[cost < 0] = 0 # hinge
                    cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                self.add_cost('IRR', self.calculate_cost(cost), bad_pairs, losses)
            if 'RII' in self.substrategy:
                try:
                    cost = distances.diag().unsqueeze(0) - distances + self.alpha.to(distances.device) # all triplets
                except AttributeError:
                    cost = distances.diag().unsqueeze(0) - distances + self.alpha
                    
                cost[cost < 0] = 0 # hinge
                cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                if self.adaptive_margin:
                    self.alpha_rii = min(max(self.min_margin, self.max_margin*(1 -  ((cost.t().numel() - (cost.t() == 0).float().sum()) / cost.t().numel()).item())), self.max_margin)
                    cost = distances.diag().unsqueeze(0) - distances + self.alpha_rii
                    cost[cost < 0] = 0 # hinge
                    cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                self.add_cost('RII', self.calculate_cost(cost.t()), bad_pairs, losses)
            if 'IRI' in self.substrategy:
                distances_image = self.dist(instance_input1, instance_input1)
                cost = distances.diag().unsqueeze(1) - distances_image + self.alpha # all triplets
                cost[cost < 0] = 0 # hinge
                cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                self.add_cost('IRI', self.calculate_cost(cost), bad_pairs, losses)
            if 'RIR' in self.substrategy:
                distances_recipe = self.dist(instance_input2, instance_input2)
                cost = distances.diag().unsqueeze(0) - distances_recipe + self.alpha # all triplets
                cost[cost < 0] = 0 # hinge
                cost[range(cost.size(0)),range(cost.size(1))] = 0 # erase pos-pos pairs
                self.add_cost('RIR', self.calculate_cost(cost), bad_pairs, losses)
            # Lifted, instance-based triplet
            if 'LIFT' in self.substrategy:
                distances_mexp = (self.alpha - distances).exp()
                sum0 = distances_mexp.sum(0)
                sum1 = distances_mexp.sum(1)
                negdiag = torch.log(sum0 + sum1 - 2*distances_mexp.diag()) # see equation 4 on the paper, this is the left side : https://arxiv.org/pdf/1511.06452.pdf
                cost = distances.diag() + negdiag
                cost[cost < 0] = 0 # hinge
                cost = cost.pow(2).sum() / 2*distances.diag().numel()
                self.add_cost('LIFT', cost, bad_pairs, losses)
            

        out = {}
        if len(bad_pairs.keys()) > 0:
            total_bad_pairs = input1.data.new([0])
            for key in bad_pairs.keys():
                total_bad_pairs += bad_pairs[key]
                out[key] = bad_pairs[key]
            total_bad_pairs = total_bad_pairs / len(bad_pairs.keys())
            out['bad_pairs'] = total_bad_pairs
        else:
            out['bad_pairs'] = input1.data.new([0])

        total_loss = input1.data.new([0])
        if len(losses.keys()) > 0:
            for key in losses.keys():
                total_loss += losses[key]
                out[key] = losses[key]
            out['loss'] = total_loss / len(losses.keys())
        else:
            out['loss'] = input1.data.new([0])
        return out

    def dist(self, input_1, input_2):
        input_1 = nn.functional.normalize(input_1)
        input_2 = nn.functional.normalize(input_2)
        return 1 - torch.mm(input_1, input_2.t())



from bootstrap.lib.options import Options
from .trijoint import Trijoint
from bootstrap.lib.logger import Logger
import click
def factory(engine=None):
    if Options()['model.name'] == 'trijoint':
        model = Trijoint(
            Options()['model'],
            Options()['dataset.nb_classes'],
            engine.dataset.keys(),
            engine)
    elif Options()['model.name'] == 'trijoint_vlp':
        model = Trijoint(
            Options()['model'],
            Options()['dataset.nb_classes'],
            ['pretrain'],
            engine)
    else:
        print(Options()['model.name'])
        raise ValueError()
        
    return model


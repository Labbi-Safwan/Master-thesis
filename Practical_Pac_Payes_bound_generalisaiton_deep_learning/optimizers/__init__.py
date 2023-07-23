from optimizers.kfac import KFACOptimizer
from optimizers.ekfac import EKFACOptimizer
from optimizers.curveball import CurveBall
from optimizers.swats import SWATS
from optimizers.padam import Padam
from optimizers.fadam import Fadam
from optimizers.lookahead import Lookahead
from optimizers.adam import Adam
from optimizers.adam_tracereg import Adam_tracereg
from optimizers.radam import RAdam


def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'ekfac':
        return EKFACOptimizer
    else:
        raise NotImplementedError
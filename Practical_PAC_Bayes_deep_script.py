import sys
sys.path.append('/home/infres/slabbi/Master_thesis/lib/python3.8/site-packages')

import torch
from laplace.baselaplace import KronLaplace
from laplace.curvature import AsdlGGN
import torch
from copy import deepcopy
import numpy as np
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.distributions import Normal
import matplotlib.pyplot as plt
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from laplace.baselaplace import KronLaplace
from laplace.curvature import AsdlGGN
from PBB.pbb.utils import runexp

"setting the parameters: to be replaced with a parser soon"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

BATCH_SIZE = 64
TRAIN_EPOCHS = 50
DELTA = 0.025
DELTA_TEST = 0.01
PRIOR = 'learnt'

SIGMAPRIOR = 0.03

PMIN = 1e-5
KL_PENALTY = 0.1
LEARNING_RATE = 0.001
MOMENTUM = 0.95
LEARNING_RATE_PRIOR = 0.005
MOMENTUM_PRIOR = 0.99

MC_SAMPLES = 1000

runexp('mnist', 'practical_bayes_deep', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=5, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2)
    



import sys
sys.path.append('/home/infres/slabbi/Master_thesis/lib/python3.8/site-packages')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from PBB.pbb.models import NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbCNNet9l, CNNet9l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, trainNNet, testNNet, Lambda_var, trainPNNet, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble, custom_weights, CNNet4l_no_activation, trainNNet_cross_entropy, testNNet_cross_entropy, nll_loss_NNet_train_set, nll_loss_NNet_test_set
from PBB.pbb.bounds import PBBobj
from PBB.pbb.utils import runexp
from PBB.pbb.data import loaddataset, loadbatches
from laplace.curvature import AsdlGGN
from laplace.baselaplace import KronLaplace

from laplace.baselaplace import KronLaplace
from laplace.curvature import AsdlGGN

"setting the parameters: to be replaced with a parser soon"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
perc_train=1.0
perc_prior=0.2
layers=9
prior_epochs=10
name_data = 'mnist'
train, test = loaddataset(name_data)
loader_kargs = {'num_workers': 1,
                'pin_memory': True} if torch.cuda.is_available() else {}
batch_size=250
learning_rate_prior=0.01
momentum_prior=0.95
verbose=False
verbose_test=False
samples_ensemble=4

L = runexp('mnist', 'practical_bayes_deep', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=device, prior_epochs=5, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2)
print(L)



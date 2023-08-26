import sys
sys.path.append('/home/infres/slabbi/Master_thesis/lib/python3.8/site-packages')
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
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

BATCH_SIZE = 64
TRAIN_EPOCHS = 10
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

# note the number of MC samples used in the paper is 150.000, which usually takes a several hours to compute
MC_SAMPLES = 100

net_classique = runexp('mnist', 'bbb', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=5, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0)

from PBB.pbb.data import loaddataset, loadbatches
from PBB.pbb.models  import computeRiskCertificates
from PBB.pbb.bounds import PBBobj
from PBB.pbb.models import NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbCNNet9l, CNNet9l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, trainNNet, testNNet, Lambda_var, trainPNNet, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble, custom_weights, CNNet4l_no_activation, trainNNet_cross_entropy, testNNet_cross_entropy, nll_loss_NNet_test_set , ProbCNNet4l_no_activation, ProbNNet2l_no_activation, NNet2l_no_activation, CNNet4l_from_PBB_net4l
from laplace.baselaplace import KronLaplace
from laplace.curvature import AsdlGGN

backend=AsdlGGN
train, test = loaddataset('mnist')
loader_kargs = {'num_workers': 1,
                'pin_memory': True} if torch.cuda.is_available() else {}
train_loader, test_loader, _, val_bound_one_batch, _, val_bound = loadbatches(
                train, test, loader_kargs, BATCH_SIZE, prior=False)
batch_size = 250
perc_train=1.0
perc_prior=0.2
learning_rate_prior=0.01
momentum_prior=0.95
prior_epochs=10
verbose=True
verbose_test=True

posterior_n_size = len(train_loader.dataset)
bound_n_size = len(val_bound.dataset)

toolarge = True
train_size = len(train_loader.dataset)
classes = len(train_loader.dataset.classes)

class CNNet4l_from_PBB_net4l(nn.Module):
    """Implementation of a standard Convolutional Neural Network with 4 layers with no activation function at the end
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self,pnet):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv1.weight = pnet.conv1.weight.mu
        self.conv1.bias = pnet.conv1.bias.mu
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv2.weight = pnet.conv2.weight.mu
        self.conv2.bias = pnet.conv2.bias.mu
        self.fc1 = nn.Linear(9216, 128)
        self.fc1.weight= pnet.fc1.weight.mu 
        self.fc1.bias= pnet.fc1.bias.mu 
        self.fc2 = nn.Linear(128, 10)
        self.fc2.weight= pnet.fc2.weight.mu 
        self.fc2.bias= pnet.fc2.bias.mu 
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

batch_size = 250
perc_train=1.0
perc_prior=0.2
learning_rate_prior=0.01
momentum_prior=0.95
prior_epochs=10
verbose=True
verbose_test=True
net0 = CNNet4l().to(DEVICE)
train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = loadbatches(
    train, test, loader_kargs, batch_size, prior=True, perc_train=perc_train, perc_prior=perc_prior)
optimizer = optim.SGD(net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
for epoch in trange(prior_epochs):
    trainNNet_cross_entropy(net0, optimizer, epoch, valid_loader,
              device=DEVICE, verbose=verbose)
errornet0 = testNNet_cross_entropy(net0, test_loader, device=DEVICE)

net1 = CNNet4l_from_PBB_net4l(net_classique).to(DEVICE)
prior_prec=1
sigma_noise=1
N = len(train_loader.dataset)
marglik_net0 = nll_loss_NNet_test_set(net0, test_loader, device = DEVICE)
marglik_net1 = nll_loss_NNet_test_set(net1, test_loader, device = DEVICE)
print('marglik_net0', marglik_net0)
print('marglik_net1', marglik_net1)
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

# We get the paramter theta star in this part
backend=AsdlGGN
net0 = CNNet4l_no_activation().to(device)
train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = loadbatches(
    train, test, loader_kargs, batch_size, prior=True, perc_train=perc_train, perc_prior=perc_prior)
optimizer = optim.SGD(net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
for epoch in trange(prior_epochs):
    trainNNet_cross_entropy(net0, optimizer, epoch, valid_loader,
              device=device, verbose=verbose)
errornet0 = testNNet_cross_entropy(net0, test_loader, device=device)

# We get here the laplace approximation of the posterior distribution and sample a (multiple) parameter(s) theta 
# from the posterior defined in the Practical PAC-Bayes generalisation bound for deep learning.

la = KronLaplace(net0, 'classification', backend=backend)
la.fit(train_loader)
theta = la.sample(n_samples = samples_ensemble)

# we create a neural netork with the same architecture as the precedent neural network but with the new weight
# obtained by sampling the posterior

new_net = custom_weights(net0, theta[0])

# We compute here some metrics

stch_err = testNNet(new_net, test_loader, device=device)
stch_loss = nll_loss_NNet_test_set(new_net , test_loader, device =device)
ens_loss, ens_err = 0.0,0.0
for index in range(samples_ensemble):
    new_net = custom_weights(new_net, theta[index]).to(device)
    ens_err += testNNet(new_net, test_loader, device=device)
    ens_loss += nll_loss_NNet_test_set(new_net , test_loader, device=device)
ens_err/= samples_ensemble
ens_loss/= samples_ensemble
print(f"***Final results***") 
print(f" Stch loss, Stch 01 error, Ens loss, Ens 01 error")
print(f" {stch_loss :.5f}, {stch_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}")




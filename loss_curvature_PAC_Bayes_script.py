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
from PBB.pbb.models import NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbCNNet9l, CNNet9l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, trainNNet, testNNet, Lambda_var, trainPNNet, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble, custom_weights, CNNet4l_no_activation, trainNNet_cross_entropy, testNNet_cross_entropy, nll_loss_NNet_train_set , ProbCNNet4l_no_activation
from PBB.pbb.bounds import PBBobj
from PBB.pbb.data import loaddataset, loadbatches
from laplace.curvature import AsdlGGN
from laplace.baselaplace import KronLaplace

from laplace.baselaplace import KronLaplace
from laplace.curvature import AsdlGGN

"setting the parameters: to be replaced with a parser soon"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

batch_size = 64
train_epochs = 10
delta = 0.025
delta_test = 0.01
prior = 'learnt'
prior_dist='gaussian'
sigma_prior = 0.03
initial_lamb=6.0
kl_penalty = 0.1
learning_rate = 0.000001
momentum = 0.95
rho_prior = math.log(math.exp(sigma_prior)-1.0)
pmin = 1e-5
mc_samples = 15
perc_train=1.0
perc_prior=0.2
layers=9
prior_epochs=10
name_data = 'mnist'
train, test = loaddataset(name_data)
loader_kargs = {'num_workers': 1,
                'pin_memory': True} if torch.cuda.is_available() else {}
objective = 'loss_curvature_bayes'
batch_size=250
learning_rate_prior=0.01
momentum_prior=0.95
verbose=True
verbose_test=True
samples_ensemble=4
model = 'cnn'
# We get the paramter theta star in this part
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

posterior_n_size = len(train_loader.dataset)
bound_n_size = len(val_bound.dataset)

toolarge = False
train_size = len(train_loader.dataset)
classes = len(train_loader.dataset.classes)

la = KronLaplace(net0, 'classification', backend=backend)
la.fit(train_loader)
net = ProbCNNet4l_no_activation(rho_prior, prior_dist=prior_dist, device=device, init_net=net0).to(device)
bound = PBBobj('loss_curvature_bayes', pmin, classes, delta,
        delta_test, mc_samples, kl_penalty, device, n_posterior = posterior_n_size, n_bound=bound_n_size ,net0 = net0, H = la.H, laplace = la)

lambda_var = Lambda_var(initial_lamb, train_size).to(device)
optimizer_lambda = optim.SGD(lambda_var.parameters(), lr=learning_rate, momentum=momentum)

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
for epoch in trange(train_epochs):
    trainPNNet(net, optimizer, bound, epoch, train_loader, lambda_var, optimizer_lambda, verbose)
    if verbose_test and ((epoch+1) % 5 == 0):
        train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge,
        bound, device=device, lambda_var=lambda_var, train_loader=val_bound, whole_train=val_bound_one_batch)

        stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
        post_loss, post_err = testPosteriorMean(net, test_loader, bound, device=device)
        ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)

        print(f"***Checkpoint results***")         
        print(f"Objective, Dataset, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_CE, Risk_01, KL, Train NLL loss, Train 01 error, Stch loss, Stch 01 error, Post mean loss, Post mean 01 error, Ens loss, Ens 01 error, 01 error prior net, perc_train, perc_prior")
        print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")

train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge, bound, device=device,
lambda_var=lambda_var, train_loader=val_bound, whole_train=val_bound_one_batch)

stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)

print(f"***Final results***") 
print(f"Objective, Dataset, Stch loss, Stch 01 error, Ens loss, Ens 01 error")
print(f"{objective}, {name_data}, {stch_loss :.5f}, {stch_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}")        




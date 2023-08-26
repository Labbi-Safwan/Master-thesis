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
from PBB.pbb.models import NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbCNNet9l, CNNet9l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, trainNNet, testNNet, Lambda_var, trainPNNet, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble, custom_weights, CNNet4l_no_activation, trainNNet_cross_entropy, testNNet_cross_entropy, cross_entropy_loss_NNet_test_set, ProbCNNet4l_no_activation, CNNet4l_from_PBB_net4l, nll_loss_NNet_test_set
from PBB.pbb.bounds import PBBobj
from PBB.pbb.data import loaddataset, loadbatches
from laplace.curvature import AsdlGGN
from laplace.baselaplace import KronLaplace

from laplace.baselaplace import KronLaplace
from laplace.curvature import AsdlGGN

"setting the parameters: to be replaced with a parser soon"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

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

batch_size = 250
train_epochs = 5
delta = 0.025
delta_test = 0.01
prior = 'learnt'
prior_dist='gaussian'
sigma_prior = 0.03
initial_lamb=6.0
kl_penalty = 0.1
learning_rate = 0.01
momentum = 0.95
rho_prior = math.log(math.exp(sigma_prior)-1.0)
pmin = 1e-5
mc_samples = 15
perc_train=1.0
perc_prior=0.2
layers=9
prior_epochs=10
beta = 10**5
name_data = 'mnist'
train, test = loaddataset(name_data)
loader_kargs = {'num_workers': 1,
                'pin_memory': True} if torch.cuda.is_available() else {}
objective = 'practical_bayes_deep'
learning_rate_prior=0.01
momentum_prior=0.95
verbose=True
verbose_test=True
samples_ensemble=4
sigma_noise = 1
model = 'cnn'

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

posterior_n_size = len(train_loader.dataset)
bound_n_size = len(val_bound.dataset)


la = KronLaplace(net0, 'classification',  prior_mean = nn.utils.parameters_to_vector(net0.parameters()) , backend=backend)
la.fit(train_loader)

samples = torch.randn(samples_ensemble, la.n_params, device=la._device)
samples = sigma_prior * la.posterior_precision.bmm(samples, exponent=-1)
theta =  la.mean.reshape(1, la.n_params) + samples.reshape(samples_ensemble, la.n_params)
# we create a neural network with the same architecture as the precedent neural network but with the new weight
# obtained by sampling the posterior

new_net = custom_weights(net0, theta[0])
errornet0 = testNNet_cross_entropy(net0, test_loader, device=device)

la = KronLaplace(new_net, 'classification',  prior_mean = nn.utils.parameters_to_vector(net0.parameters()),sigma_noise=sigma_noise,
                 prior_precision=perc_prior,   backend=backend)
la.fit(train_loader)

toolarge = False
train_size = len(train_loader.dataset)
classes = len(train_loader.dataset.classes)

bound = PBBobj('practical_bayes_deep', pmin, classes, delta,
        delta_test, mc_samples, kl_penalty, device, n_posterior = posterior_n_size, n_bound=bound_n_size , theta = theta[0], laplace = la)



net = ProbCNNet4l_no_activation(rho_prior, prior_dist=prior_dist, device=device, init_net=net0).to(device)

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
for epoch in trange(train_epochs):
    trainPNNet(net, optimizer, bound, epoch, train_loader, beta= beta)
# We compute here some metrics

stch_err = testNNet_cross_entropy(new_net, test_loader, device=device)
stch_loss = cross_entropy_loss_NNet_test_set(new_net , test_loader, device =device)
ens_loss, ens_err = 0.0,0.0
for index in range(samples_ensemble):
    new_net = custom_weights(new_net, theta[index]).to(device)
    ens_err += testNNet_cross_entropy(new_net, test_loader, device=device)
    ens_loss += cross_entropy_loss_NNet_test_set(new_net , test_loader, device=device)
ens_err/= samples_ensemble
ens_loss/= samples_ensemble
print(f"***Final results***") 
print(f" Stch loss, Stch 01 error, Ens loss, Ens 01 error")
print(f" {stch_loss :.5f}, {stch_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}")

train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge, bound, device=DEVICE,
                                                                                        lambda_var=1, train_loader=val_bound,
                                                                                        whole_train=val_bound_one_batch)

print(objective, train_obj)
print('risk',risk_01)



from copy import deepcopy
import numpy as np
import torch
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
from PBB.pbb.data import loaddataset, loadbatches

from laplace.baselaplace import KronLaplace
from laplace.curvature import AsdlGGN


GB_FACTOR = 1024 ** 3




def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])




def get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device):
    log_prior_prec_init = np.log(prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    log_prior_prec.requires_grad = True
    return log_prior_prec




def valid_performance(model, test_loader, likelihood, criterion, device):
    N = len(test_loader.dataset)
    perf = 0
    nll = 0
    for X, y in test_loader:
        X, y = X.detach().to(device), y.detach().to(device)
        with torch.no_grad():
            f = model(X)
        if likelihood == 'classification':
            perf += (torch.argmax(f, dim=-1) == y).sum() / N
        elif likelihood == 'heteroscedastic_regression':
            perf += (y.squeeze() + 0.5 * f[:, 0] / f[:, 1]).square().sum() / N
        else:
            perf += (f - y).square().sum() / N
        nll += criterion(f, y) / len(test_loader)
    return perf.item(), nll.item()




def get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min):
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        return ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        return CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')




def get_model_optimizer(optimizer, model, lr, weight_decay=0):
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        # fixup parameters should have 10x smaller learning rate
        is_fixup = lambda param: param.size() == torch.Size([1])  # scalars
        fixup_params = [p for p in model.parameters() if is_fixup(p)]
        standard_params = [p for p in model.parameters() if not is_fixup(p)]
        params = [{'params': standard_params}, {'params': fixup_params, 'lr': lr / 10.}]
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')




def gradient_to_vector(parameters):
    return parameters_to_vector([e.grad for e in parameters])




def vector_to_gradient(vec, parameters):
    return vector_to_parameters(vec, [e.grad for e in parameters])




def marglik_optimization(model,
                         train_loader,
                         valid_loader=None,
                         likelihood='classification',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         sigma_noise_init=1.,
                         temperature=1.,
                         n_epochs=50,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='cos',
                         n_epochs_burnin=0,
                         n_hypersteps=100,
                         marglik_frequency=1,
                         lr_hyp=1e-1,
                         lr_hyp_min=1e-1,
                         laplace=KronLaplace,
                         backend=AsdlGGN,
                         early_stopping=False):
    """Runs marglik optimization training for a given model and training dataloader.


    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    likelihood : str
        'classification', 'regression', 'heteroscedastic_regression'
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    sigma_noise_init : float
        initial observation noise (for regression only)
    temperature : float
        factor for the likelihood for 'overcounting' data.
        Often required when using data augmentation.
    n_epochs : int
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'adam' or 'sgd'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int
        how many steps to take on the hyperparameters when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
    lr_hyp : float
        learning rate for hyperparameters (should be between 1e-3 and 1)
    laplace : Laplace
        type of Laplace approximation (Kron/Diag/Full)
    backend : Backend
        AsdlGGN/AsdlEF or BackPackGGN/BackPackEF
    stochastic_grad : bool
    independent : bool
        whether to use independent functional laplace
    single_output : bool
        whether to use single random output for functional laplace
    kron_jac : bool
        whether to use kron_jac in the backend


    Returns
    -------
    lap : Laplace
        lapalce approximation
    model : torch.nn.Module
    margliks : list
    losses : list
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    best_model_dict = None


    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)
    hyperparameters.append(log_prior_prec)


    # set up loss (and observation noise hyperparam)
    if likelihood == 'classification':
        criterion = CrossEntropyLoss(reduction='mean')
        sigma_noise = 1
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)
    else:
        raise ValueError()


    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)


    n_steps = ((n_epochs - n_epochs_burnin) // marglik_frequency) * n_hypersteps
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)
    hyper_scheduler = CosineAnnealingLR(hyper_optimizer, n_steps, eta_min=lr_hyp_min)


    losses = list()
    valid_perfs = list()
    valid_nlls = list()
    margliks = list()
    best_marglik = np.inf


    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        epoch_nll = 0
        epoch_log = dict(epoch=epoch)


        # standard NN training per batch
        torch.cuda.empty_cache()
        for X, y in train_loader:
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()


            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = 1 / temperature / (2 * sigma_noise.square())
            else:
                crit_factor = 1 / temperature
            prior_prec = torch.exp(log_prior_prec).detach()
            delta = expand_prior_precision(prior_prec, model)


            f = model(X)


            theta = parameters_to_vector(model.parameters())
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()


            epoch_loss += loss.cpu().item() / len(train_loader)
            epoch_nll += criterion(f.detach(), y).item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            elif likelihood == 'heteroscedastic_regression':
                epoch_perf += (y.squeeze() + 0.5 * f[:, 0] / f[:, 1]).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()


        losses.append(epoch_loss)
        logging.info(f'MARGLIK[epoch={epoch}]: train. perf={epoch_perf:.2f}; loss={epoch_loss:.5f}; nll={epoch_nll:.5f}')
        optimizer.zero_grad(set_to_none=True)
        llr = scheduler.get_last_lr()[0]
        epoch_log.update({'train/loss': epoch_loss, 'train/nll': epoch_nll, 'train/perf': epoch_perf, 'train/lr': llr})
        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                if likelihood == 'regression':
                    def val_criterion(f, y):
                        assert f.shape == y.shape
                        log_lik = Normal(loc=f, scale=sigma_noise).log_prob(y)
                        return -log_lik.mean()
                else:
                    val_criterion = criterion
                val_perf, val_nll = valid_performance(model, valid_loader, likelihood, val_criterion, device)
                valid_perfs.append(val_perf)
                valid_nlls.append(val_nll)
                logging.info(f'MARGLIK[epoch={epoch}]: valid. perf={val_perf:.2f}; nll={val_nll:.5f}.')
                epoch_log.update({'valid/perf': val_perf, 'valid/nll': val_nll})


        # only update hyperparameters every "Frequency" steps after "burnin"
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue


        # 1. fit laplace approximation
        torch.cuda.empty_cache()


        sigma_noise = 1 if likelihood != 'regression' else torch.exp(log_sigma_noise)
        prior_prec = torch.exp(log_prior_prec)
        lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                        temperature=temperature, backend=backend)
        lap.fit(train_loader)
        # first optimize prior precision jointly with direct marglik grad
        margliks_local = list()
        for i in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            sigma_noise = None if likelihood != 'regression' else torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise) / N
            marglik.backward()
            margliks_local.append(marglik.item())
            hyper_optimizer.step()
            hyper_scheduler.step()


        marglik = margliks_local[-1]


        if likelihood == 'regression':
            epoch_log['hyperparams/sigma_noise'] = torch.exp(log_sigma_noise.detach()).cpu().item()
        epoch_log['train/marglik'] = marglik
        margliks.append(marglik)
        del lap


        # early stopping on marginal likelihood
        if early_stopping and (margliks[-1] < best_marglik):
            best_model_dict = deepcopy(model.state_dict())
            best_precision = deepcopy(prior_prec.detach())
            best_sigma = 1 if likelihood != 'regression' else deepcopy(sigma_noise.detach())
            best_marglik = margliks[-1]


    if early_stopping and (best_model_dict is not None):
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
    else:
        sigma_noise = 1 if sigma_noise is None else sigma_noise


    lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                  temperature=temperature, backend=backend)
    lap.fit(train_loader)
    return lap, model, margliks
    
import torch
import logging
import torch.nn as nn
from PBB.pbb.models import ProbNNet4l, NNet4l, CNNet4l, trainNNet_cross_entropy , testNNet_cross_entropy
import torchvision
import math

class CNNet4l_from_PBB_net4l_soft(nn.Module):
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

net0 = CNNet4l_from_PBB_net4l_soft(net)
prior_prec=1
sigma_noise=1
N = len(train_loader.dataset)
marglik_net0 = nll_loss_NNet_test_set(net0, test_loader, device = DEVICE)
print('marglik_net0', marglik_net0)




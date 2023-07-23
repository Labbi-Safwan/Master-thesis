"""
This module defines a linear operator to compute the hessian-vector product
for a given pytorch model using subsampled data.
"""

# Modified by Xingchen Wan in Sep 2019 to accommodate our code base

import torch
from eigenthings.power_iter import deflated_power_iteration
from eigenthings.lanczos import lanczos


class Operator:
    """
    maps x -> Lx for a linear operator L
    """

    def __init__(self, size):
        self.size = size

    def apply(self, vec):
        """
        Function mapping vec -> L vec where L is a linear operator
        """
        raise NotImplementedError


class LambdaOperator(Operator):
    """
    Linear operator based on a provided lambda function
    """
    def __init__(self, apply_fn, size):
        super(LambdaOperator, self).__init__(size)
        self.apply_fn = apply_fn

    def apply(self, x):
        return self.apply_fn(x)


class HVPOperator(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    max_samples: max number of examples per batch using all GPUs.
    """

    def __init__(self, model, dataloader, criterion, use_gpu=True,
                 full_dataset=False, max_samples=256,
                 bn_train_mode=True):
        size = int(sum(p.numel() for p in model.parameters()))
        super(HVPOperator, self).__init__(size)
        self.grad_vec = torch.zeros(size)
        self.model = model
        if use_gpu:
            self.model = self.model.cuda()
        self.dataloader = dataloader
        # Make a copy since we will go over it a bunch
        self.dataloader_iter = iter(dataloader)
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.full_dataset = full_dataset
        self.max_samples = max_samples
        self.bn_train_mode = bn_train_mode

    def apply(self, vec):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        # compute original gradient, tracking computation graph
        self.model.eval()
        if self.bn_train_mode:
            self.model.apply(self._bn_train_mode)

        self.zero_grad()
        if self.full_dataset:
            grad_vec = self.prepare_full_grad()
        else:
            grad_vec = self.prepare_grad()
        self.zero_grad()
        # take the second gradient
        grad_grad = torch.autograd.grad(grad_vec, self.model.parameters(),
                                        grad_outputs=vec,
                                        only_inputs=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1)
                                      for g in grad_grad])
        return hessian_vec_prod

    def zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def prepare_full_grad(self):
        """
        Compute gradient w.r.t loss over all parameters, where loss
        is computed over the full dataloader
        """
        grad_vec = None
        n = len(self.dataloader)
        for _ in range(n):
            batch_grad = self.prepare_grad()
            if grad_vec is not None:
                grad_vec += batch_grad
            else:
                grad_vec = batch_grad
        self.grad_vec = grad_vec / n
        return self.grad_vec

    def prepare_grad(self):
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        try:
            all_inputs, all_targets = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            all_inputs, all_targets = next(self.dataloader_iter)

        num_chunks = max(1, len(all_inputs) // self.max_samples)

        grad_vec = None

        input_chunks = all_inputs.chunk(num_chunks)
        target_chunks = all_targets.chunk(num_chunks)
        for input, target in zip(input_chunks, target_chunks):
            if self.use_gpu:
                input = input.cuda()
                target = target.cuda()

            output = self.model(input)
            loss = self.criterion(output, target)
            grad_dict = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True)
            if grad_vec is not None:
                grad_vec += torch.cat([g.contiguous().view(-1) for g in grad_dict])
            else:
                grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
        grad_vec /= num_chunks
        self.grad_vec = grad_vec
        return self.grad_vec

    @staticmethod
    def _bn_train_mode(module):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.train()


def compute_hessian_eigenthings(model, dataloader, loss,
                                num_eigenthings=10,
                                full_dataset=True,
                                mode='power_iter',
                                use_gpu=True,
                                max_samples=512,
                                **kwargs):
    """
    Computes the top `num_eigenthings` eigenvalues and eigenvecs
    for the hessian of the given model by using subsampled power iteration
    with deflation and the hessian-vector product

    Parameters
    ---------------

    model : Module
        pytorch model for this netowrk
    dataloader : torch.data.DataLoader
        dataloader with x,y pairs for which we compute the loss.
    loss : torch.nn.modules.Loss | torch.nn.functional criterion
        loss function to differentiate through
    num_eigenthings : int
        number of eigenvalues/eigenvecs to compute. computed in order of
        decreasing eigenvalue magnitude.
    full_dataset : boolean
        if true, each power iteration call evaluates the gradient over the
        whole dataset.
    mode : str ['power_iter', 'lanczos']
        which backend to use to compute the top eigenvalues.
    use_gpu:
        if true, attempt to use cuda for all lin alg computatoins
    max_samples:
        the maximum number of samples that can fit on-memory. used
        to accumulate gradients for large batches.
    **kwargs:
        contains additional parameters passed onto lanczos or power_iter.
    """
    hvp_operator = HVPOperator(model, dataloader, loss,
                               use_gpu=use_gpu,
                               full_dataset=full_dataset,
                               max_samples=max_samples)
    eigenvals, eigenvecs = None, None
    if mode == 'power_iter':
        eigenvals, eigenvecs = deflated_power_iteration(hvp_operator,
                                                        num_eigenthings,
                                                        use_gpu=use_gpu,
                                                        **kwargs)
    elif mode == 'lanczos':
        eigenvals, eigenvecs = lanczos(hvp_operator,
                                       num_eigenthings,
                                       use_gpu=use_gpu,
                                       **kwargs)
    else:
        raise ValueError("Unsupported mode %s (must be power_iter or lanczos)"
                         % mode)
    return eigenvals, eigenvecs

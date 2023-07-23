import argparse

import os
import numpy as np

import torch

from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG

parser = argparse.ArgumentParser(description='SGD_noschedule/SWA training')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT', required=True,
                    help='checkpoint to load model (default: None)')
parser.add_argument('--swag', action='store_true')

parser.add_argument('--iters', type=int, default=300, metavar='N', help='number of lanczos steps (default: 20)')
parser.add_argument('--num_samples', type=int, default=None, metavar='N', help='number of data points to use (default: the whole dataset)')
parser.add_argument('--subsample_seed', type=int, default=None, metavar='N', help='random seed for dataset subsamling (default: None')

parser.add_argument('--bn_train_mode_off', action='store_true')
parser.add_argument('--cov_grad', action='store_true')

parser.add_argument('--grad_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to save gradient (default: None)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
   args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

if args.dataset == 'CIFAR10':
    dataset_class = data.CIFAR10AUG
    num_classes = 10
elif args.dataset == 'CIFAR100':
    dataset_class = data.CIFAR100AUG
    num_classes = 100
else:
    dataset_class = None
assert dataset_class is not None

dataset = dataset_class(
    root=os.path.join(args.data_path, args.dataset.lower()),
    train=True,
    transform=model_cfg.transform_test,
    download=True,
    shuffle_seed=args.subsample_seed,
)

if args.num_samples is not None:
    dataset = torch.utils.data.Subset(dataset, torch.arange(args.num_samples))


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

full_datasets, _ = data.datasets(
    args.dataset,
    args.data_path,
    transform_train=model_cfg.transform_train,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
)

full_loader = torch.utils.data.DataLoader(
    full_datasets['train'],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

print(len(loader.dataset))


print('Preparing model')
print(*model_cfg.args, dict(**model_cfg.kwargs))
if not args.swag:
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    print('Loading %s' % args.ckpt)
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['state_dict'])
else:
    swag_model = SWAG(model_cfg.base,
                 subspace_type='random',
                 *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    print('Loading %s' % args.ckpt)
    checkpoint = torch.load(args.ckpt)
    swag_model.load_state_dict(checkpoint['state_dict'], strict=False)
    swag_model.set_swa()
    model = swag_model.base_model

model.to(args.device)

num_parametrs = sum([p.numel() for p in model.parameters()])

criterion = losses.cross_entropy

utils.bn_update(full_loader, model)
print(utils.eval(loader, model, criterion))

if args.cov_grad:
    grad = utils.covgrad(loader, model, criterion, cuda=args.device.type == 'cuda', bn_train_mode=False).cpu().numpy()
else:
    grad = utils.grad(loader, model, criterion, cuda=args.device.type == 'cuda', bn_train_mode=False).cpu().numpy()


np.savez(
    args.grad_path,
    grad=grad
)
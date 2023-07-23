import argparse
import torch
from eigenthings.hvp_operator import compute_hessian_eigenthings
from curvature import data, models, losses, utils


parser = argparse.ArgumentParser(description='EigenthingsDemo')
parser.add_argument('--num_eigenthings', default=5, type=int,help='number of eigenvals/vecs to compute')
parser.add_argument('--batch_size', default=128, type=int, help='train set batch size')
parser.add_argument('--eval_batch_size', default=16, type=int, help='test set batch size')
parser.add_argument('--momentum', default=0.0, type=float,help='power iteration momentum term')
parser.add_argument('--num_steps', default=20, type=int,help='number of power iter steps')
parser.add_argument('--cuda', action='store_true', help='if true, use CUDA/GPUs')
parser.add_argument('--dir', type=str, default=None, required=True,
                    help='training directory (default: None)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')

args = parser.parse_args()
print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
)

lanczos_ds, _ = data.datasets(
    args.dataset,
    args.data_path,
    transform_train=model_cfg.transform_test,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
    train_subset=args.batch_size,
    train_subset_seed=1,
)

lanczos_loader = torch.utils.data.DataLoader(
    lanczos_ds['train'],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

print('Preparing model')
print(*model_cfg.args, dict(**model_cfg.kwargs))
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

criterion = torch.nn.CrossEntropyLoss()
eigenvals, eigenvecs = compute_hessian_eigenthings(model, lanczos_loader,
                                                       criterion,
                                                       args.num_eigenthings,
                                                       args.num_steps,
                                                       momentum=args.momentum,
                                                       use_gpu=True)
print("Eigenvecs:")
print(eigenvecs)
print("Eigenvals:")
print(eigenvals)
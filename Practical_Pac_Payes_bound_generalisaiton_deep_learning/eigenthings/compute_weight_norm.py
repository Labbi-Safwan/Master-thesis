import torch
import argparse
from curvature import data, models
import numpy as np
from curvature.methods.swag import SWAG

parser = argparse.ArgumentParser(description="Weight Norm Computation")
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset name (default: CIFAR100)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument('--dir',  type=str, default=None, required=True)
parser.add_argument("--save_path", type=str, default=None, required=True)
parser.add_argument("--swag", action='store_true')

args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
   args.device = torch.device('cuda')
else:
   args.device = torch.device('cpu')


def prep_model():
    model_cfg = getattr(models, args.model)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    datasets, num_classes = data.datasets(
        args.dataset,
        args.data_path,
        transform_train=model_cfg.transform_test,
        transform_test=model_cfg.transform_test,
        use_validation=True,
    )

    full_datasets, _ = data.datasets(
        args.dataset,
        args.data_path,
        transform_train=model_cfg.transform_train,
        transform_test=model_cfg.transform_test,
        use_validation=True,
    )

    print('Preparing model')
    print(*model_cfg.args, dict(**model_cfg.kwargs))
    if not args.swag:
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    else:
        model = SWAG(model_cfg.base, subspace_type='random', *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    return model


def compute_weight(file_path, model):
    print('Loading %s' % file_path)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['state_dict'])

    if args.swag:
        model.set_swa()
    model.to(args.device)

    w = torch.cat([param.detach().cpu().view(-1) for param in model.parameters()])
    l2_norm = torch.norm(w).numpy()
    linf_norm = torch.norm(w, float('inf')).numpy()
    return l2_norm, linf_norm


model = prep_model()
x = np.arange(0, args.epochs, 1)
files = ['checkpoint-0000'+str(x_)+".pt" if x_ < 10 else 'checkpoint-000'+str(x_)+".pt" if x_<100 else 'checkpoint-00'+str(x_)+".pt"  for x_ in x]
l2_norms = np.full(len(files), np.nan)
linf_norms = np.full(len(files), np.nan)

for i in range(len(files)):
    try:
        l2_norms[i], linf_norms[i] = compute_weight(args.dir+files[i], model)
    except FileNotFoundError:
        l2_norms[i], linf_norms[i] = np.nan, np.nan
np.savez(
    args.save_path,
    l2_norms=l2_norms,
    linf_norms=linf_norms
)
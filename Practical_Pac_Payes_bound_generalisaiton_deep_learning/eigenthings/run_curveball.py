import argparse
import os
import sys
import time
import tabulate

import torch

from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG, SWA
import numpy as np
from optimizers import CurveBall

parser = argparse.ArgumentParser(description='CurveBall/SWA training')
parser.add_argument('--dir', type=str, default="/home/xwan/PycharmProjects/kfac-curvature/out/VGG16BN/CurveBall/run1",
                    required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume  from (default: None)')
# parser.add_argument('--resume', type=str, default="/home/xwan/PycharmProjects/kfac-curvature/out/VGG16BN/SGD_noschedule/run1/checkpoint-00153.pt", metavar='CKPT',
#                     help='checkpoint to resume training from (default: None)')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD_noschedule momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')
parser.add_argument('--step_schedule', action='store_true')
parser.add_argument("--step_schedule_r", type=int, default=50, help="frequency in epoch of reducing the learning rate")
parser.add_argument("--step_schedule_d", type=float, default=0.5, help='size of the learning rate step reduction. '
                                                                       '(default 0.5: i.e. halving the learning rate)')

parser.add_argument('--swag', action='store_true')
# Xingchen Addition
parser.add_argument("--no_covariance", action='store_true', help='Do not use Gaussian covariance in SWAG - essentially use SWA')

parser.add_argument('--swag_subspace', choices=['covariance', 'pca', 'freq_dir'], default='pca')
parser.add_argument('--swag_rank', type=int, default=20, help='SWAG covariance rank')
parser.add_argument('--swag_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02). Note that ')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--swag_save_stats', action='store_true', help='Save the SWA statistics')
parser.add_argument('--swag_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')

parser.add_argument("--auto_lambda_factor", type=float, default=0.999, help='Curveball: the lambda adjustment factor. The author recommended 0.999.')
parser.add_argument("--auto_lambda_interval", type=int, default=5 ,help="Curveball: the interval of automatic lambda adjustment")
parser.add_argument('--lr_init', type=float, default=-1.,help="Curveball: the initial value of learning rate. Set to a value below 0 for auto parameter"
                                                              " calibration. Under automatica parameter tuning, this will always be 1.")
parser.add_argument("--beta_init", type=float, default=-1.,
                    help="Curveball: the initial value of beta. Set to a value below 0 for auto parameter calibration")
parser.add_argument("--rho_init", type=float, default=-1, help='Curveball: initial value of rho i.e. momentum. Set to a value below 0 for auto parameter calibration')
parser.add_argument("--lambda_init", type=float, default=10., help='Curveball: init value of lambda. For BN networks, the authors recommended the value 1 where for non-BN, 10.')
parser.add_argument("--gamma_upper_limit", type=float, default=1.5, help='Curveball: upper limit of gamma, above which automatic lambda adjustment will be triggered.')
parser.add_argument("--gamma_lower_limit", type=float, default=0.5, help="Curveball: lower limit of gamma, below which automatic lambda adjustment will be triggered.")

# Xingchen Addition
parser.add_argument("--save_freq_weight_norm", type=int, default=1, metavar='N', help='save frequency of weight norm')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

if args.no_schedule and args.step_schedule:
    raise ValueError("Both no_schedule and step_schedule are turned on. Quitting due to ambiguity.")

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
)

print('Preparing model')
print(*model_cfg.args, dict(**model_cfg.kwargs))
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

if args.swag:
    if not args.no_covariance:
        print('SWA-Gaussian training')
        swag_model = SWAG(model_cfg.base,
                      subspace_type=args.swag_subspace, subspace_kwargs={'max_rank': args.swag_rank},
                      *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        swag_model.to(args.device)
    else:
        print('No Covariance Estimation')
        swag_model = SWA(model_cfg.base, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        swag_model.to(args.device)
else:
    print('Curveball training')


if args.no_schedule:
    print("Constant learning rate schedule")
elif args.step_schedule:
    print("Piecewise constant learning rate schedule. Decaying " + str(args.step_schedule_d * 100)
          + "% every " + str(args.step_schedule_r) + " epochs.")
else:
    print("Linearly decaying learning rate schedule")


def schedule(epoch):
    """Use a linearly decaying learning rate"""
    t = epoch / (args.swag_start if args.swag else args.epochs)
    lr_ratio = args.swag_lr / args.lr_init if args.swag else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


def schedule_piecewise_const(epoch):
    """
    Use a piecewise constant learning rate
    d: The proportion of new learning rate to the initial learning rate. 0.5 means halving the learning rate
    r: frequency of reducing the learning rate. e.g. 40: reducing learning rate every 40 epochs
    """
    r = args.step_schedule_r
    d = args.step_schedule_d
    factor = d ** np.floor((1 + epoch) / r)
    return args.lr_init * factor


criterion = losses.cross_entropy_func

#optimizer = Curveball(
#    model.parameters(),
#    criterion,
#    lr=args.lr_init,
#    weight_decay=args.wd,
#    auto_hyperparam=args.auto_hyperparam,
#    cuda=torch.cuda.is_available(),
#    rho=args.rho_init,
#    beta=args.beta_init,
#    param_lambda=args.lambda_init,
#    auto_beta_rho_interval=args.auto_beta_rho_interval,
#    auto_lambda_factor=args.auto_lambda_factor,
#    auto_lambda_interval=args.auto_lambda_interval,
#    gamma_lower_limit=args.gamma_lower_limit,
#    gamma_upper_limit=args.gamma_upper_limit,
#)

optimizer = CurveBall(model.parameters(),
                      beta=args.beta_init,
                      momentum=args.rho_init,
                      lambd=args.lambda_init,
                      lambda_factor=args.auto_lambda_factor,
                      lambda_low=args.gamma_lower_limit,
                      lambda_high=args.gamma_upper_limit,
                      lambda_interval=args.auto_lambda_interval,
                      weight_decay=args.wd)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if args.swag and args.swag_resume is not None:
    checkpoint = torch.load(args.swag_resume)
    swag_model.load_state_dict(checkpoint['state_dict'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'mem_usage']
if args.swag:
    columns = columns[:-2] + ['swa_te_loss', 'swa_tr_acc', 'swa_te_acc'] + columns[-2:]
    swag_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    epoch=start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict()
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    # END ADD NEW CODE DIEGO
    if not args.no_schedule:
        if args.step_schedule:
            lr = schedule_piecewise_const(epoch)
        else:
            lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    train_res = utils.train_epoch_adaptive(loaders['train'], model, criterion, optimizer, verbose=False)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None}

    if args.swag and (epoch + 1) > args.swag_start and (epoch + 1 - args.swag_start) % args.swag_c_epochs == 0:
        swag_model.collect_model(model)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swag_model.set_swa()
            utils.bn_update(loaders['train'], swag_model)
            train_res_swag = utils.train_epoch(loaders['train'], swag_model, criterion, optimizer)
            swag_res = utils.eval(loaders['test'], swag_model, criterion)
        else:
            swag_res = {'loss': None, 'accuracy': None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        if args.swag and (epoch + 1) > args.swag_start:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name='swag',
                epoch=epoch + 1,
                state_dict=swag_model.state_dict(),
            )

        # Xingchen code addition - enable to save the L2 and Linf weight norms
    if (epoch + 1) % args.save_freq_weight_norm == 0:
        utils.save_weight_norm(
            args.dir,
            epoch + 1,
            name='weight_norm',
            model=model
        )
        if args.swag and (epoch + 1) > args.swag_start:
            utils.save_weight_norm(
                args.dir,
                epoch + 1,
                name='swa_weight_norm',
                model=swag_model
            )

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'],
              time_ep, memory_usage]

    np.savez(
        args.dir + 'stats-' + str(epoch),
        train_loss=train_res['loss'],
time_ep = time_ep,
memory_usage = memory_usage,

        train_accuracy=train_res['accuracy'],
        test_loss=test_res['loss'],
        test_accuracy=test_res['accuracy'],
    )

    if args.swag:
        values = values[:-2] + [swag_res['loss'],  train_res_swag['accuracy'], swag_res['accuracy']] + values[-2:]
        np.savez(
            args.dir + 'stats-' + str(epoch),
        train_loss=train_res['loss'],
time_ep = time_ep,
memory_usage = memory_usage,

            train_accuracy=train_res['accuracy'],
            test_loss=test_res['loss'],
            test_accuracy=test_res['accuracy'],
            swag_loss=swag_res['loss'],
            swag_train_acc = train_res_swag['accuracy'],
            swag_accuracy=swag_res['accuracy']
        )

    if args.swag:
        values = values[:-2] + [swag_res['loss'],  train_res_swag['accuracy'], swag_res['accuracy']] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        epoch=args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
    if args.swag and args.swag_save_stats:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            name='swag',
            epoch=args.epochs,
            state_dict=swag_model.state_dict(),
        )

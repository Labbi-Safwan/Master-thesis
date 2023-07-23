import argparse
import os
import sys
import time
import tabulate
import numpy as np

import torch
import torch.optim

from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG, SWA
from optimizers import KFACOptimizer
import pdb


parser = argparse.ArgumentParser(description='SGD_noschedule/SWA training')
parser.add_argument('--dir', type=str, default="/home/xwan/PycharmProjects/kfac-curvature/out/VGG16BN/kfac-run2/", required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default="/home/xwan/PycharmProjects/kfac-curvature/data", required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default="VGG16", required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=150, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD_noschedule momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 1e-4)')
parser.add_argument("--decoupled_wd", action='store_true', help='whether to use decoupled weight decay')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')

parser.add_argument('--swag', action='store_true')
parser.add_argument("--no_covariance", action='store_true', help='Do not use Gaussian covariance in SWAG - essentially use SWA')
parser.add_argument('--swag_subspace', choices=['covariance', 'pca', 'freq_dir'], default='pca')
parser.add_argument('--swag_rank', type=int, default=20, help='SWAG covariance rank')
parser.add_argument('--swag_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--stat_decay', default=0.95, type=float)
parser.add_argument('--damping', default=100, type=float)
parser.add_argument('--ma', action='store_true', help=f'whether to use moving average of learned damping or just learned damping')
parser.add_argument('--kl_clip', default=0.01, type=float)
parser.add_argument('--TCov', default=10, type=int)
parser.add_argument('--TScal', default=10, type=int)
parser.add_argument('--TInv', default=100, type=int)

parser.add_argument('--adaptive', action='store_true', help='Whether use the adaptive damping adjustment, adaptive scaling '
                                                                 'of proposal and factored Tikhonov regularisation.')
parser.add_argument('--TAdapt', type=int, default=5, help='Frequency of automatic damping adjustment (default: 5)')
parser.add_argument('--omega', type=float, default=19./20., help='Scaling factor per iteration for damping adjustment')

parser.add_argument('--train_first_epoch_with_sgd', action='store_true')
parser.add_argument('--verbose', action='store_true', help='whether to use verbose mode for training.')
# Xingchen Addition
parser.add_argument("--save_freq_weight_norm", type=int, default=1, metavar='N', help='save frequency of weight norm')

#Diego addition
parser.add_argument('--num_samples', type=int, default=None, metavar='N', help='number of data points to use (default: the whole dataset)')
parser.add_argument('--subsample_seed', type=int, default=None, metavar='N', help='random seed for dataset subsamling (default: None')
parser.add_argument('--wd_freq', type=int, default=5, metavar='N', help='wd update frequency (default: 5)')
parser.add_argument('--stats_batch', type=int, default=512, metavar='B', help='batch size used to compute the statistics (default: 1)')
parser.add_argument('--wd_start', type=int, default=1, metavar='M', help='number of iterations to run before doing adaptive damping (default: 20)')
#this option does not exist for KFAC
#parser.add_argument('--wd_mode_off', action='store_true', help='turn off wd mode of the regularizer')
parser.add_argument('--clip_alpha', type=float, default=0.001, metavar='N', help='clip alpha (default: 0.01)')
parser.add_argument('--curvature_matrix', type=str, default="hessian", help='curvature matrix GN or Hessian')



args = parser.parse_args()

args.dir = args.dir +'/'+ args.dataset+'/'+args.model + '/shrinkage_KFAC'+'/seed='+str(args.seed)+'_lr='+str(args.lr_init)+'_damping='+str(args.damping)+'_stats_batch='+str(args.stats_batch)+'_wd='+str(args.wd)+'_decoupled_wd='+str(args.decoupled_wd)+'_ma='+str(args.ma)+'_schedule='+str(args.no_schedule)+'_wd_freq='+str(args.wd_freq)+'_wdstart='+str(args.wd_start)+'_matrix='+str(args.curvature_matrix)+'/'

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

datasets, num_classes = data.datasets(
    args.dataset,
    args.data_path,
    transform_train=model_cfg.transform_test,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
    train_subset=args.num_samples,
    train_subset_seed=args.subsample_seed,
)


stats_loader = torch.utils.data.DataLoader(
    datasets['train'],
    batch_size=args.stats_batch,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
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
    print('KFAC training')

print('***Model Hyperparameters***')
print('Initial Learning Rate', args.lr_init)
print('Weight Decay:', args.wd)
print('Damping', args.damping)
print('Momentum', args.momentum)
print('Number of Epochs', args.epochs)
print('Dataset', args.dataset)

if args.adaptive:
    criterion = losses.cross_entropy_func
else:
    criterion = losses.cross_entropy

optimizer = KFACOptimizer(model,
                          lr=args.lr_init,
                          momentum=args.momentum,
                          stat_decay=args.stat_decay,
                          damping=args.damping,
                          kl_clip=args.kl_clip,
                          weight_decay=args.wd,
                          TCov=args.TCov,
                          Tadapt=args.TAdapt,
                          omega=args.omega,
                          adaptive_mode=args.adaptive,
                          decoupled_wd=args.decoupled_wd,
                          TInv=args.TInv)
sgd_optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=args.lr_init,
#     betas=[0.9, 0.9],
#     eps=1e-05,
#     weight_decay=args.wd,
#     amsgrad=False
#     )

# optimizer = torch.optim.Adadelta(
#     model.parameters(),
#     lr=1,
#     rho=0.9,
#     eps=1e-06,
#     weight_decay=args.wd)

#All Adam Variants do not work
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

#optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#RMSprop also does not work
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

columns = ['ep', 'lr', 'damping', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'te_top5_acc', 'time', 'mem_usage']

if args.swag:
    columns = columns[:-2] + ['swa_te_loss', 'swa_te_acc', 'swa_te_top5_acc'] + columns[-2:]
    swag_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    epoch=start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict()
)


def schedule(epoch):
    """
    Learning rate scheduler - taper learning rate as a function of number of epochs
    :param epoch:
    :return:
    """
    t = epoch / (args.swag_start if args.swag else args.epochs)
    lr_ratio = args.swag_lr / args.lr_init if args.swag else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor
damping = args.damping
alpha_ma = 1
for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    lr = schedule(epoch)
    # Adjust learning rate
    if not args.no_schedule:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init
    if epoch % args.wd_freq == 0 and epoch>args.wd_start:
        #print('learning damping')
        num_batches = len(loaders['train'])
        loss_stats = utils.loss_stats(loaders['train'], model, criterion, cuda=True, bn_train_mode=False, curvature_matrix=args.curvature_matrix)
        hess_var = loss_stats['hess_var'].item()
        delta = loss_stats['delta'].item()

        alpha = 1.0 - args.stats_batch * hess_var / len(stats_loader.dataset) / delta
        print(alpha)

        alpha_ma = 0.5 * alpha_ma + (1.0 - 0.5) * alpha
        mu = max(0.0, loss_stats['hess_mu'].item())
        if args.ma:
            wd = (1.0 - alpha_ma) / alpha_ma * mu
        else:
            wd = (1.0 - alpha) / alpha * mu
        damping = (1.0 - alpha) / alpha + lr
        print('damping = '+str(damping))

        # for param_group in optimizer.param_groups:
        #     param_group['alpha'] = alpha_ma
        #     param_group['mu'] = mu
        # pdb.set_trace()
        utils.adjust_kfac_damping(optimizer, damping)
        # print('kfac damping adjusted')

        # optimizer = KFACOptimizer(model,
        #                           lr=lr,
        #                           momentum=args.momentum,
        #                           stat_decay=args.stat_decay,
        #                           damping=damping,
        #                           kl_clip=args.kl_clip,
        #                           weight_decay=args.wd,
        #                           TCov=args.TCov,
        #                           Tadapt=args.TAdapt,
        #                           omega=args.omega,
        #                           adaptive_mode=args.adaptive,
        #                           decoupled_wd=args.decoupled_wd,
        #                           TInv=args.TInv)
        # del loss_stats
        # torch.cuda.empty_cache()

    if epoch == 0 and args.train_first_epoch_with_sgd:
        train_res = utils.train_epoch(loaders['train'], model, criterion, sgd_optimizer, verbose=args.verbose)
        print("Training the first epoch with SGD_noschedule")
    elif args.adaptive:
        train_res = utils.train_epoch_adaptive(loaders['train'], model, criterion, optimizer, verbose=args.verbose)
    else:
        train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, verbose=args.verbose)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

    if args.swag and (epoch + 1) >= args.swag_start and (epoch + 1 - args.swag_start) % args.swag_c_epochs == 0:
        swag_model.collect_model(model)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swag_model.set_swa()
            utils.bn_update(loaders['train'], swag_model)
            swag_res = utils.eval(loaders['test'], swag_model, criterion)
        else:
            swag_res = {'loss': None, 'accuracy': None, "top5_accuracy": None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )

        if args.swag:
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
    memory_usage = torch.cuda.memory_allocated()/(1024.0 ** 3)

    values = [epoch + 1, lr, damping, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'],
              test_res['top5_accuracy'], time_ep, memory_usage]
    np.savez(
        args.dir + 'stats-' + str(epoch),
        train_loss=train_res['loss'],
        time_ep = time_ep,
        memory_usage = memory_usage,
        train_accuracy=train_res['accuracy'],
        train_top5_accuracy = train_res['top5_accuracy'],
        test_loss=test_res['loss'],
        test_accuracy=test_res['accuracy'],
        test_top5_accuracy = test_res['top5_accuracy'],
        damping = damping
    )

    if args.swag:
        values = values[:-2] + [swag_res['loss'], swag_res['accuracy'], swag_res['top5_accuracy']] + values[-2:]
        np.savez(
            args.dir + 'stats-' + str(epoch),
            train_loss=train_res['loss'],
            time_ep = time_ep,
            memory_usage = memory_usage,
            train_accuracy=train_res['accuracy'],
            train_top5_accuracy = train_res['top5_accuracy'],
            test_loss=test_res['loss'],
            test_accuracy=test_res['accuracy'],
            test_top5_accuracy = test_res['top5_accuracy'],
            swag_loss=swag_res['loss'],
            swag_accuracy=swag_res['accuracy'],
            swag_top5_accuracy = swag_res['top5_accuracy'],
            damping = damping
        )

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

    if np.isnan(train_res['loss']):
        break
    if args.dataset == 'CIFAR100':
        if train_res['accuracy'] < 5 & epoch > 10:
            break
    elif args.dataset == 'CIFAR10':
        if train_res['accuracy'] < 20 & epoch > 10:
            break



    # Tensorboard visualisation
#    if tb is not None:
#        tb.add_scalar("data/train_loss", train_res['loss'], epoch)
#        tb.add_scalar("data/train_acc", train_res['accuracy'], epoch)
#        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
#            tb.add_scalar("data/test_loss", test_res['loss'], epoch)
#            tb.add_scalar("data/test_acc", test_res['accuracy'], epoch)

#if tb is not None:
#    tb.close()

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        epoch=args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
    if args.swag:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            name='swag',
            epoch=args.epochs,
            state_dict=swag_model.state_dict(),
        )

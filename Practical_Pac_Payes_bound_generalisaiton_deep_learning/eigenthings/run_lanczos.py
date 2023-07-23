import argparse
import os
import sys
import time
import tabulate

import torch
import numpy as np

from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG
from optimizers.lanczosopt import LanczosSGD

parser = argparse.ArgumentParser(description='SGD_noschedule/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True,
                    help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--matrix_type', type=str, default='hessian', metavar='matrix_type',
                    help='type of curvature matrix used Hessian or GN')

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N',
                    help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N',
                    help='evaluation frequency (default: 5)')

parser.add_argument('--lr_init', type=float, default=1, metavar='LR',
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD_noschedule momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--lanczos_steps', type=int, default=10, help='number of lanczos steps (default: 10)')
parser.add_argument('--lanczos_batch', type=int, default=128, help='batch size for hessian vector product(default: 256)')
parser.add_argument('--lanczos_lr', type=float, default=1, help='LR for lanczosopt (default: 1)')
parser.add_argument('--lanczos_beta', type=float, default=100, help='beta for lanczosopt (default: 1)')

parser.add_argument('--swag', action='store_true')
parser.add_argument('--swag_subspace', choices=['covariance', 'pca', 'freq_dir'], default='pca')
parser.add_argument('--swag_rank', type=int, default=10, help='SWAG covariance rank')
parser.add_argument('--swag_start', type=float, default=161, metavar='N',
                    help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.02, metavar='LR',
                    help='SWA LR (default: 0.02)')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
args.dir = args.dir +'/'+ args.dataset+'/'+args.model + '/Lanczos/lr='+str(args.lr_init)+'_matrix='+args.matrix_type+'_damping='+str(args.lanczos_beta)+'_wd='+str(args.wd)+'_batch_size='+str(args.batch_size)+'/'

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

lanczos_ds, _ = data.datasets(
    args.dataset,
    args.data_path,
    transform_train=model_cfg.transform_test,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
    train_subset=args.lanczos_batch,
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

if args.swag:
    print('Lanczos training')
    swag_model = SWAG(model_cfg.base,
                      subspace_type=args.swag_subspace,
                      subspace_kwargs={'max_rank': args.swag_rank},
                      *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swag_model.to(args.device)
else:
    print('Lanczos training')


def schedule(epoch):
    t = epoch / (args.swag_start if args.swag else args.epochs)
    lr_ratio = args.swag_lr / args.lr_init if args.swag else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


criterion = losses.cross_entropy


class HessVecProduct(object):
    def __init__(self, loader, model, criterion, move_output2cpu=False):
        self.loader = loader
        self.model = model
        self.criterion = criterion
        self.iters = 0
        self.timestamp = time.time()
        self.move_output2cpu = move_output2cpu

    def __call__(self, vector):
        start_time = time.time()
        if args.matrix_type == 'hessian':
            output = utils.hess_vec(vector, self.loader, self.model, self.criterion, cuda=args.device.type=='cuda')
        elif args.matrix_type == 'gn':
            output = utils.gn_vec(vector, self.loader, self.model, self.criterion, cuda=args.device.type=='cuda')
        time_diff = time.time() - start_time

        self.iters += 1
        #print('Iter %d. Time: %.2f' % (self.iters, time_diff))
        if self.move_output2cpu:
            return output.cpu().unsqueeze(1)
        return output.unsqueeze(1)


productor = HessVecProduct(lanczos_loader, model, criterion)

lanczos_optimizer = LanczosSGD(
    model.parameters(),
    hess_vec_closure=productor,
    lanczos_steps=args.lanczos_steps,
    beta=args.lanczos_beta,
    lr=args.lanczos_lr,
    weight_decay=args.wd
)

sgd_optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

optimizer = lanczos_optimizer
print('using lanczos to start')

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'mem_usage']
if args.swag:
    columns = columns[:-2] + ['swa_te_loss', 'swa_tr_acc', 'swa_te_acc'] + columns[-2:]
    swag_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    epoch=start_epoch,
    state_dict=model.state_dict(),
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.swag or args.swag and epoch < args.swag_start:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)

    if args.swag and epoch + 1 == args.swag_start:
        optimizer = lanczos_optimizer
        print('using lanczos')

    # train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, verbose=args.swag and epoch + 1 >= args.swag_start)
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, verbose=False)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None}


    if args.swag and (epoch + 1) >= args.swag_start and (
            epoch + 1 - args.swag_start) % args.swag_c_epochs == 0:
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

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

    values = [epoch + 1, args.lr_init, train_res['loss'], train_res['accuracy'], test_res['loss'],
              test_res['accuracy'],
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
    if np.isnan(train_res['loss']):
        print('game over')
        break
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
    )
    if args.swag:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            name='swag',
            epoch=args.epochs,
            state_dict=swag_model.state_dict(),
        )

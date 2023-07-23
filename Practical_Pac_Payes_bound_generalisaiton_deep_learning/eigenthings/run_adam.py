import argparse
import os
import sys
import time

import numpy as np
import tabulate
import torch

from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG, SWA
from optimizers.adam import Adam

print('numpy imported')

parser = argparse.ArgumentParser(description='SGD_noschedule/SWA training')
parser.add_argument('--dir', type=str, default="/home/xwan/PycharmProjects/kfac-curvature/out/VGG16BN/Adam/run1",
                    required=False, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
# parser.add_argument('--resume', type=str, default="/home/xwan/PycharmProjects/kfac-curvature/out/VGG16BN/Adam/run1/checkpoint-00075.pt", metavar='CKPT',
# help='checkpoint to resume training from (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.001, metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD_noschedule momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--decoupled_wd', action='store_true', help="Enable to use AdamW - decoupled weight decay")
parser.add_argument("--normalized_wd", action='store_true',
                    help='Whether to use normalised wd. WD = WD_norm \sqrt(\frac{b}{BT})')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')

parser.add_argument('--swag', action='store_true')

# Xingchen Addition
parser.add_argument("--no_covariance", action='store_true',
                    help='Do not use Gaussian covariance in SWAG - essentially use SWA')

parser.add_argument('--swag_subspace', choices=['covariance', 'pca', 'freq_dir'], default='pca')
parser.add_argument('--swag_rank', type=int, default=20, help='SWAG covariance rank')
parser.add_argument('--swag_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.0005, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--swag_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--verbose', action='store_true', help='Whether to use verbose mode trianing')
# Xingchen Addition
parser.add_argument("--save_freq_weight_norm", type=int, default=1, metavar='N', help='save frequency of weight norm')

args = parser.parse_args()

add = ''
end = ''
if args.decoupled_wd:
    if not args.swag:
        add = 'W'
    else:
        add = 'X'
        end = '_swa_start=' + str(args.swag_start) + '_swa_lr=' + str(args.swag_lr) + ''
elif args.swag:
    add = "SWA"

args.dir = args.dir +'/'+ args.dataset+'/'+args.model+'/Adam'+add+'/seed='+str(args.seed)+'_lr='+str(args.lr_init)+'_wd='+str(args.wd)+end+'/'

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
    print('Adam training')


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

if args.normalized_wd:
    weight_decay = args.wd * np.sqrt(args.batch_size / (args.epochs * len(loaders['train'])))
else:
    weight_decay = args.wd

optimizer = Adam(
    model.parameters(),
    lr=args.lr_init,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
    amsgrad=False,
    decoupled_wd=args.decoupled_wd
)

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

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'te_top5_acc', 'time', 'mem_usage']
if args.swag:
    columns = columns[:-2] + ['swa_tr_loss', 'swa_tr_acc',
                              'swa_te_loss', 'swa_te_acc', 'swa_te_top5_acc'] + columns[-2:]
    swag_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

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
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, verbose=args.verbose)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

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
              test_res['top5_accuracy'], time_ep, memory_usage]

    np.savez(
        args.dir + 'stats-' + str(epoch),
        train_loss=train_res['loss'],
        time_ep=time_ep,
        memory_usage=memory_usage,
        train_accuracy=train_res['accuracy'],
        train_top5_accuracy=train_res['top5_accuracy'],
        test_loss=test_res['loss'],
        test_accuracy=test_res['accuracy'],
        test_top5_accuracy=test_res['top5_accuracy']
    )
    if args.swag and (epoch + 1) > args.swag_start and (epoch + 1 - args.swag_start) % args.swag_c_epochs == 0:
        swag_model.collect_model(model)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swag_model.set_swa()
            utils.bn_update(loaders['train'], swag_model)
            train_res_swag = utils.train_epoch(loaders['train'], swag_model, criterion, optimizer)
            swag_res = utils.eval(loaders['test'], swag_model, criterion)
        else:
            swag_res = {'loss': None, 'accuracy': None, "top5_accuracy": None}
            train_res_swag = {'loss': None, 'accuracy': None}

        np.savez(
            args.dir + 'stats-' + str(epoch),
            train_loss=train_res['loss'],
            time_ep=time_ep,
            memory_usage=memory_usage,
            train_accuracy=train_res['accuracy'],
            train_top5_accuracy=train_res['top5_accuracy'],
            test_loss=test_res['loss'],
            test_accuracy=test_res['accuracy'],
            test_top5_accuracy=test_res['top5_accuracy'],
            swag_loss=swag_res['loss'],
            swag_train_acc=train_res_swag['accuracy'],
            swag_accuracy=swag_res['accuracy'],
            swag_top5_accuracy=swag_res['top5_accuracy']
        )
    else:
        train_res_swag = {'loss': None, 'accuracy': None}

    if args.swag:
        values = values[:-2] + [train_res_swag['loss'], train_res_swag['accuracy'],
                                swag_res['loss'], swag_res['accuracy'], swag_res['top5_accuracy']] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

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

    # if np.isnan(test_res['loss']):
    #     break

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        epoch=args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
#    if args.swag:
#        utils.save_checkpoint(
#            args.dir,
#            args.epochs,
#            name='swag',
#            epoch=args.epochs,
#            state_dict=swag_model.state_dict(),
#        )

# Xingchen Wan | 23 December 2019
# Runs recurrent neural network / LSTM model on a character level language model task (i.e. predicting the next chara
# cter, etc)

# This is a quick integration of the existing codes into our codebase, some basic functionality are lacking (e.g.
# saving models and resuming training, SWA model saving and etc).

import argparse
import os
import tabulate
import sys
import time
from curvature.models import *
from curvature.methods.swag import SWA
from curvature.data_nlp import LanguageDataLoader, check_data_preprocessing

from optimizers.adam import Adam

from curvature.utils_language import *
from curvature.utils import adjust_learning_rate
import logging

# logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='NLP Task')
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument('--redo_preprocess', action='store_true')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seq-length', default=64, type=int)

parser.add_argument('--model', type=str, required=True)
parser.add_argument('--num-layers', default=2, type=int)
parser.add_argument('--embedding-dim', default=128, type=int)
parser.add_argument('--hidden-dim', default=128, type=int)
parser.add_argument('--zoneout', default=0, type=float)
parser.add_argument('--dropout', default=0, type=float)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency')

# Optimizer related settings
parser.add_argument('--lr_init', type=float, default=0.002, metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--optimizer', default='Adam', type=str)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--decoupled_wd', action='store_true')
parser.add_argument('--grad_clip', default=5, type=float)

parser.add_argument('--no_schedule', action='store_true')

# SWA/SWAG related settings
parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_start', type=int, default=161)
parser.add_argument('--swa_lr', type=float, default=0.02, help='SWA LR')
parser.add_argument('--verbose', action='store_true')
parser.add_argument("--seed", type=int)

args = parser.parse_args()

if args.dir[-1] != "/" : args.dir += "/"
args.dir = args.dir + args.dataset + '/'+args.model + '/' + args.optimizer
if args.swa:
   args.dir += "_swa"
if args.decoupled_wd:
    args.dir += "_W"
args.dir += '/seed='+str(args.seed)+'_lr='+str(args.lr_init)+'_wd='+str(args.wd)+'/'

args.device = None

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


json_path, h5_path = check_data_preprocessing(dataset=args.dataset, path=args.data_path, flush=args.redo_preprocess)
loader = LanguageDataLoader(
    filename=h5_path,
    batch_size=args.batch_size,
    seq_length=args.seq_length
)

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model = LanguageModel()
model.load_tokendata(json_path)
model.build_model(
  layertype=args.model,
  dropout=args.dropout,
  num_layers=args.num_layers,
  D=args.embedding_dim,
  H=args.hidden_dim,
  zoneout=args.zoneout
  )
print(model.layers)

if args.verbose:
    print('By-layer summary of the model built:')
    print(model.layers)
model.to(args.device)

if args.swa:
    swa_model = SWA(model)
    swa_model.to(args.device)

assert args.optimizer in ['SGD', 'Adam']


if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=args.wd
    )
else:
    optimizer = Adam(
        model.parameters(),
        lr=args.lr_init,
        betas=(args.momentum, 0.999),
        weight_decay=args.wd,
        decoupled_wd=args.decoupled_wd
    )

criterion = nn.CrossEntropyLoss()


# Linear learning rate schedule
def schedule(epoch):
    t = epoch / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


start_epoch = 0
columns = ['ep', 'lr', 'tr_loss', 'te_loss', 'time', 'mem_usage']
if args.swa:
    columns = columns[:-2] + ['swa_te_loss'] + columns[-2:]
    swa_res = {'loss': None, }

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    if not args.no_schedule:
        lr = schedule(epoch)
        adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init
    train_res = train_epoch(model, loader, criterion, optimizer, epoch, args.seq_length, device=args.device,
                            grad_clip=args.grad_clip)
    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = eval_epoch(model, loader, criterion, device=args.device)
    else:
        test_res = {'loss': None}
    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated()/(1024.0 ** 3)
    values = [epoch + 1, lr, train_res['loss'], test_res['loss'], time_ep, memory_usage]

    np.savez(
        args.dir + 'stats-' + str(epoch),
        train_loss=train_res['loss'],
        time_ep=time_ep,
        memory_usage=memory_usage,
        test_loss=test_res['loss'],
    )

    if args.swa and (epoch + 1) > args.swa_start:
        swa_model.collect_model(model)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swa_model.set_swa()
            # note that the BN line is omitted here - LSTM/RNN models used here do not have BN
            train_res_swa = train_epoch(swa_model, loader, criterion, optimizer, epoch, args.seq_length,
                                        device=args.device, grad_clip=args.grad_clip)
            swa_res = eval_epoch(swa_model, loader, criterion,  device=args.device,)
        else:
            swa_res = {'loss': None}
            train_res_swa = {'loss': None}
        values = values[:-2] + [swa_res['loss']] + values[-2:]
        np.savez(
            args.dir + 'stats-' + str(epoch),
            train_loss=train_res['loss'],
            time_ep=time_ep,
            memory_usage=memory_usage,
            test_loss=test_res['loss'],
            swag_loss=swa_res['loss'],
        )
    else:
        swa_res = {'loss': None}
        train_res_swa = {'loss': None}

    if args.swa:
        values = values[:-2] + [swa_res['loss']] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

    if (epoch + 1) % args.save_freq == 0:
        model.save_model(
            "%s_%d" % (args.dir, epoch)
        )
        # todo: save model for the swa models

if args.epochs % args.save_freq != 0:
    model.save_model(
        "%s_%d" % (args.dir, args.epochs)
    )

import torch
from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG
from optimizers import KFACOptimizer
import os, time
import tabulate
import numpy as np

class Scheduler:
    def __init__(self, optimizer,
                 dir,
                 model,
                 dataset: str ='CIFAR100',
                 data_path: str ="/home/xwan/PycharmProjects/kfac-curvature/data",
                 batch_size: int = 128,
                 epochs: int = 300,
                 save_freq: int = 25,
                 lr_init: float = 0.1,
                 momentum: float = 0.9,
                 wd: float = 5e-4,
                 no_schedule: bool = False,
                 swag: bool = False,
                 swag_subspace: str = "pca",
                 swag_lr = 0.02,
                 swag_rank: int = 20,
                 swag_start: int = 161,
                 swag_c_epochs: int = 1,
                 seed: int = 1,
                 stat_decay: float = 0.95,
                 damping: float = 0.01,
                 kl_clip: float = 1e-2,
                 KFAC_TCov: int = 10,
                 KFAC_TScal: int = 10,
                 KFAC_TInv: int = 100
                 ):
        if optimizer is None:
            self.optimizer = "SGD_noschedule"
            print("Using default SGD_noschedule optimizer")
        self.dir = dir
        self.model = model
        self.dataset = dataset
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_freq = save_freq
        self.lr_init = lr_init
        self.momentum = momentum
        self.wd = wd
        self.no_schedule = no_schedule
        self.swag = swag
        self.swag_lr = swag_lr
        self.swag_subspace = swag_subspace
        self.swag_rank = swag_rank
        self.swag_start = swag_start
        self.swag_c_epochs = swag_c_epochs
        self.seed = seed
        self.stat_decay = stat_decay
        self.damping = damping
        self.kl_clip = kl_clip
        self.KFAC_TCov = KFAC_TCov
        self.KFAC_TScal = KFAC_TScal
        self.KFAC_TInv = KFAC_TInv

        self.eval_freq = 25
        # CUDA availability
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def print_attributes(self):
        print("Current attribute of the scheduler")
        print(self.__dict__)

    def run_job(self):
        if self.optimizer == "SGD_noschedule":
            pass

        elif self.optimizer == 'KFAC':
            print("KFAC Optimisation")


        elif self.optimizer == 'adam':
            pass

        else:
            raise ValueError("The specified optimizer type is unsupported.")

    def prepare(self):
        print('Preparing directory %s' % self.dir)
        os.makedirs(self.dir, exist_ok=True)

        torch.backends.cudnn.benchmark = True
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        print('Using model %s' % self.model)
        model_cfg = getattr(models, self.model)

        loaders, num_classes = data.loaders(
            self.dataset,
            self.data_path,
            self.batch_size,
            self.num_workers,
            model_cfg.transform_train,
            model_cfg.transform_test,
            use_validation=True,
        )

        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        model.to(self.device)

        if self.swag:
            print('SWAG is enabled')
            swag_model = SWAG(model_cfg.base,
                              subspace_type=self.swag_subspace, subspace_kwargs={'max_rank': self.swag_rank},
                              *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            swag_model.to(self.device)

        criterion = losses.cross_entropy
        # criterion = nn.CrossEntropyLoss()

        optim_obj = KFACOptimizer(model,
                                  lr=self.lr_init,
                                  momentum=self.momentum,
                                  stat_decay=self.stat_decay,
                                  damping=self.damping,
                                  kl_clip=self.kl_clip,
                                  weight_decay=self.wd,
                                  TCov=self.KFAC_TCov,
                                  TInv=self.KFAC_TInv)
        if args.swag:
            columns = columns[:-2] + ['swa_te_loss', 'swa_tr_acc', 'swa_te_acc'] + columns[-2:]
            swag_res = {'loss': None, 'accuracy': None}

        return optim_obj, criterion, loaders

    def lr_scheduler(self, epoch):
        t = epoch / (self.swag_start if self.swag else self.epochs)
        lr_ratio = self.swag_lr / self.lr_init if self.swag else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return self.lr_init * factor

    def run(self, optim_obj, criterion, loaders):
        for epoch in range(0, self.epochs):
            time_ep = time.time()

            # Adjust learning rate
            if not self.no_schedule:
                lr = self.lr_scheduler(epoch)
                utils.adjust_learning_rate(optim_obj, lr)
            else:
                lr = self.lr_init

            train_res = utils.train_epoch(loaders['train'], self.model, criterion, optim_obj, verbose=False)

            if epoch == 0 or epoch % self.eval_freq == self.eval_freq - 1 or epoch == self.epochs - 1:
                test_res = utils.eval(loaders['test'], self.model, criterion)
            else:
                test_res = {'loss': None, 'accuracy': None}

            if self.swag and (epoch + 1) >= self.swag_start and (epoch + 1 - self.swag_start) % self.swag_c_epochs == 0:
                self.swag_model.collect_model(self.model)
                if epoch == 0 or epoch % self.eval_freq == self.eval_freq - 1 or epoch == self.epochs - 1:
                    self.swag_model.set_swa()
                    utils.bn_update(loaders['train'], self.swag_model)
                    swag_res = utils.eval(loaders['test'], self.swag_model, criterion)
                else:
                    swag_res = {'loss': None, 'accuracy': None}

            if (epoch + 1) % self.save_freq == 0:
                utils.save_checkpoint(
                    self.dir,
                    epoch + 1,
                    epoch=epoch + 1,
                    state_dict=self.model.state_dict(),
                    optimizer=optim_obj.state_dict()
                )

                if self.swag:
                    utils.save_checkpoint(
                        self.dir,
                        epoch + 1,
                        name='swag',
                        epoch=epoch + 1,
                        state_dict=self.swag_model.state_dict(),
                    )

            time_ep = time.time() - time_ep
            memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

            values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'],
                      time_ep, memory_usage]
            np.savez(
                self.dir + 'stats-' + str(epoch),
                train_loss=train_res['loss'],
                train_accuracy=train_res['accuracy'],
                test_loss=test_res['loss'],
                test_accuracy=test_res['accuracy'],
            )

            if self.swag:
                values = values[:-2] + [swag_res['loss'],  train_res_swag['accuracy'], swag_res['accuracy']] + values[-2:]
                np.savez(
                    self.dir + 'stats-' + str(epoch),
                    train_loss=train_res['loss'],
                    train_accuracy=train_res['accuracy'],
                    test_loss=test_res['loss'],
                    test_accuracy=test_res['accuracy'],
                    swag_loss=swag_res['loss'],
            swag_train_acc = train_res_swag['accuracy'],
                    swag_accuracy=swag_res['accuracy']
                )
            columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'mem_usage']

            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
            if epoch % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)
if __name__  == "__main__":
    pass
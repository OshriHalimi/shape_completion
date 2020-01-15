# -*- coding: utf-8 -*-
import torch
import torch.nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Utils:
import re
import os
import glob
import time
import math

from util.gen import Progbar, banner
import Config as cfg
from Config import DATA as dat
from Config import NET as net


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------


class NeuralNet:
    # This wrapper presumes we are dealing with a:
    # 1. PytorchNet object
    # 2. Input arguments to constructor are: self.device, dat.num_classes(), dat.input_channels(), dat.shape()
    def __init__(self, resume=True, ckp_name_prefix=None):

        # Decide on device:
        if torch.cuda.is_available():
            # print('CUDA FOUND!')
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            if torch.cuda.device_count() > 1:
                raise NotImplementedError
                # This line enables multiple GPUs, but changes the layer names a bit
                # self.net = torch.nn.DataParallel(self.net)
                #  Useful if you have multiple GPUs - does not hurt otherwise
        else:
            self.device = torch.device('cpu')
            # torch.set_num_threads(4) # Presuming 4 cores
            print('WARNING: Found no valid GPU device - Running on CPU')
        # Build Model:
        print(f'==> Building model {net.__name__} on the dataset {dat.name()}')
        self.net = net(self.device, dat.num_classes(), dat.input_channels(), dat.shape())
        print(f'==> Detected family model of {self.net.family_name()}')

        if resume:
            print(f'==> Resuming from checkpoint via sorting method: {cfg.RESUME_METHOD}')
            assert os.path.isdir(cfg.CHECKPOINT_DIR), 'Error: no checkpoint directory found!'

            ck_file = self.__class__.resume_methods[cfg.RESUME_METHOD](self.net.family_name(),
                                                                       ckp_name_prefix=ckp_name_prefix)
            if ck_file is None:
                print(f'-E- Found no valid checkpoints for {net.__name__} on {dat.name()}')
                self.best_val_acc = 0
                self.start_epoch = 0
            else:
                checkpoint = torch.load(ck_file, map_location=self.device)
                self._load_checkpoint(checkpoint['net'])
                self.best_val_acc = checkpoint['acc']
                self.start_epoch = checkpoint['epoch']
                assert (dat.name() == checkpoint['dataset'])

                print(f'==> Loaded model with val-acc of {self.best_val_acc:.3f}')

        else:
            self.best_val_acc = 0
            self.start_epoch = 0

        self.net = self.net.to(self.device,non_blocking=True) # MANO - Added non_blocking

        # Build SGD Algorithm:
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_gen, self.val_gen, self.classes = (None, None, None)
        self.optimizer = None

    def train(self, epochs, lr=0.1, set_size=None, batch_size=cfg.BATCH_SIZE, ckp_name_prefix=None):
        if ckp_name_prefix is not None:
            self.best_val_acc = 0

        (self.train_gen, set_size), (self.val_gen, _) = dat.trainset(batch_size=batch_size, max_samples=set_size)
        print(f'==> Training on {set_size} samples with batch size of {batch_size} and lr = {lr}')

        if cfg.SGD_METHOD == 'Nesterov':
            self.optimizer = optim.SGD(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr, momentum=0.9,
                                       weight_decay=5e-4)
        elif cfg.SGD_METHOD == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
        else:
            raise NotImplementedError
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=cfg.N_EPOCHS_TO_WAIT_BEFORE_LR_DECAY)

        p = Progbar(epochs)
        t_start = time.time()
        batches_per_step = math.ceil(set_size / batch_size)
        for epoch in range(self.start_epoch, self.start_epoch + epochs):

            if cfg.VERBOSITY > 0:
                banner(f'Epoch: {epoch}')
                t_step_start = time.time()
            train_loss, train_acc, train_count = self._train_step()
            val_loss, val_acc, val_count = self.test(self.val_gen)
            self.scheduler.step(val_loss)
            if cfg.VERBOSITY > 0:
                t_step_end = time.time()
                batch_time = round((t_step_end - t_step_start) / batches_per_step, 3)
                p.add(1, values=[("t_loss", train_loss), ("t_acc", train_acc), ("v_loss", val_loss), ("v_acc", val_acc),
                                 ("batch_time", batch_time), ("lr", self.optimizer.param_groups[0]['lr'])])
            else:
                p.add(1,
                      values=[("t_loss", train_loss), ("t_acc", train_acc), ("v_loss", val_loss), ("v_acc", val_acc),
                              ("lr", self.optimizer.param_groups[0]['lr'])])
            self._checkpoint(val_acc, epoch + 1, ckp_name_prefix=ckp_name_prefix)
        t_end = time.time()
        print(f'==> Total train time: {t_end - t_start:.3f} secs :: per epoch: {(t_end - t_start) / epochs:.3f} secs')
        banner('Training Phase - End')

    def test(self, data_gen, print_it=False):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        top5_count = [0]  # Add more elements here if you want more ks
        with torch.no_grad():  # TODO - Fix the odd intialize spatial layers bug
            for batch_idx, (inputs, targets) in enumerate(data_gen):
                inputs, targets = inputs.to(self.device,non_blocking=True), targets.to(self.device,non_blocking=True)
                # MANO - Added non_blocking
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # TODO - Export this out of the function - Replace the above code with a call to the following and topk=(1,5)
                top5_count = [sum(x) for x in zip(top5_count, self._accuracy(outputs, targets,
                                                                             topk=(5,)))]

                # val_loss = f'{test_loss/(batch_idx+1):.3f}.'
                # acc = f'{100.*correct/total:.3f}.'
                # count = f'{correct}/{total}'

        top1_acc = 100. * correct / total
        top5_acc = 100. * top5_count[0] / total
        top1_count = f'{correct}/{total}'
        top5_count = f'{top5_count[0]}/{total}'

        if print_it:
            print(
                f'==> Asserted Top1 test-acc of: {top1_acc:.3f}% [{top1_count}]. Top5 is {top5_acc:.3f}% [{top5_count}]')
        return test_loss, top1_acc, top1_count

    def summary(self, x_size, print_it=True):
        return self.net.summary(x_size, print_it=print_it)

    def print_weights(self):
        self.net.print_weights()

    def output_size(self, x_shape, cuda_allowed=True):
        return self.net.output_size(x_shape, cuda_allowed)

    def _checkpoint(self, val_acc, epoch, ckp_name_prefix=None):

        # Decide on whether to checkpoint or not:
        save_it = val_acc > self.best_val_acc
        if save_it and cfg.DONT_SAVE_REDUNDANT:
            #            target = os.path.join(cfg.CHECKPOINT_DIR, f'{self.net.family_name()}_{dat.name()}_*_ckpt.t7')
            #            checkpoints = [os.path.basename(f) for f in glob.glob(target)]
            #            if ckp_name_prefix is not None:
            #                target = os.path.join(cfg.CHECKPOINT_DIR, f'{self.net.family_name()}_{dat.name()}_*_ckpt_{ckp_name_prefix}.t7')
            #                checkpoints += [os.path.basename(f) for f in glob.glob(target)]
            if ckp_name_prefix is None:
                target = os.path.join(cfg.CHECKPOINT_DIR, f'{self.net.family_name()}_{dat.name()}_*_ckpt.t7')
            else:
                target = os.path.join(cfg.CHECKPOINT_DIR,
                                      f'{self.net.family_name()}_{dat.name()}_*_ckpt_{ckp_name_prefix}.t7')
            checkpoints = [os.path.basename(f) for f in glob.glob(target)]
            if checkpoints:
                best_cp_val_acc = max(
                    [float(f.replace(f'{self.net.family_name()}_{dat.name()}', '').split('_')[1]) for f in checkpoints])
                if best_cp_val_acc >= val_acc:
                    save_it = False
                    print(f'\nResuming without save - Found valid checkpoint with higher val_acc: {best_cp_val_acc}')
        # Do checkpoint
        val_acc = round(val_acc, 3)  # Don't allow too long a number
        if save_it:
            print(f'\nBeat val_acc record of {self.best_val_acc} with {val_acc} - Saving checkpoint')
            state = {
                'net': self.net.state_dict(),
                'dataset': dat.name(),
                'acc': val_acc,
                'epoch': epoch,
            }
            if not os.path.isdir(cfg.CHECKPOINT_DIR):
                os.mkdir(cfg.CHECKPOINT_DIR)

            if ckp_name_prefix is None:
                cp_name = f'{self.net.family_name()}_{dat.name()}_{val_acc}_ckpt.t7'
            else:
                cp_name = f'{self.net.family_name()}_{dat.name()}_{val_acc}_ckpt_{ckp_name_prefix}.t7'
            torch.save(state, os.path.join(cfg.CHECKPOINT_DIR, cp_name))
            self.best_val_acc = val_acc

    def _train_step(self):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        if cfg.VERBOSITY > 0:
            prog_batch = Progbar(len(self.train_gen))
        for batch_idx, (inputs, targets) in enumerate(self.train_gen):
            # Training step
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Collect results:
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if cfg.VERBOSITY > 0:
                prog_batch.add(1, values=[("t_loss", train_loss / (batch_idx + 1)), ("t_acc", 100. * correct / total)])

        total_acc = 100. * correct / total
        count = f'{correct}/{total}'
        return train_loss, total_acc, count

    def _load_checkpoint(self, loaded_dict, optional_fill=('.*\.num_batches_tracked',),
                         total_ignore=('.*pred\.', '.*pred2\.', '.*pred1\.', 'features\..*\.conv_filt\.weight')):

        # Make a regex that matches if any of our regexes match.
        opt_fill = "(" + ")|(".join(optional_fill) + ")"
        tot_ignore = "(" + ")|(".join(total_ignore) + ")"

        curr_dict = self.net.state_dict()
        filtered_dict = {}

        for k, v in loaded_dict.items():
            if not re.match(tot_ignore, k):  # If in ignore list, ignore
                if k in curr_dict:
                    filtered_dict[k] = v
                else:  # Check if it is possible to ignore it being gone
                    if not re.match(opt_fill, k):
                        assert False, f'Fatal: found unknown entry {k} in loaded checkpoint'

        assert filtered_dict, 'State dictionary is empty'
        # Also check for missing entries in loaded checkpoint
        for k, v in curr_dict.items():
            if k not in loaded_dict and not (re.match(opt_fill, k) or re.match(tot_ignore, k)):
                assert False, f'Fatal: missing entry {k} from checkpoint'

        # Overwrite entries in the existing state dict
        curr_dict.update(filtered_dict)
        self.net.load_state_dict(curr_dict)

    @staticmethod
    def _accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        list_correct = []
        for k in topk:
            list_correct.append(int(correct[:k].view(-1).float().sum(0)))
        return list_correct

    @staticmethod
    def _find_top_val_acc_checkpoint(family_name, ckp_name_prefix):
        if ckp_name_prefix is None:
            target = os.path.join(cfg.CHECKPOINT_DIR, f'{family_name}_{dat.name()}_*_ckpt.t7')
        else:
            target = os.path.join(cfg.CHECKPOINT_DIR, f'{family_name}_{dat.name()}_*_ckpt_{ckp_name_prefix}.t7')
        checkpoints = [os.path.basename(f) for f in glob.glob(target)]
        if not checkpoints:
            return None
        else:
            checkpoints.sort(key=lambda x: float(x.replace(f'{family_name}_{dat.name()}', '').split('_')[1]))
            # print(checkpoints)
            return os.path.join(cfg.CHECKPOINT_DIR, checkpoints[-1])

    @staticmethod
    def _find_latest_checkpoint(family_name, ckp_name_prefix):
        if ckp_name_prefix is None:
            target = os.path.join(cfg.CHECKPOINT_DIR, f'{family_name}_{dat.name()}_*_ckpt.t7')
        else:
            target = os.path.join(cfg.CHECKPOINT_DIR, f'{family_name}_{dat.name()}_*_ckpt_{ckp_name_prefix}.t7')
        checkpoints = glob.glob(target)
        if not checkpoints:
            return None
        else:
            checkpoints.sort(key=os.path.getmtime)
            # print(checkpoints)
            return checkpoints[-1]

    # Static Variables
    resume_methods = {
        'Time': _find_latest_checkpoint.__func__,
        'ValAcc': _find_top_val_acc_checkpoint.__func__
    }

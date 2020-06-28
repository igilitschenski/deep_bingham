import time
import torch

from modules import maad
from utils import AverageMeter


class Trainer(object):
    """ Trainer for Bingham Orientation Uncertainty estimation.

    Arguments:
        device (torch.device): The device on which the training will happen.

    """
    def __init__(self, device, floating_point_type="float"):
        self._device = device
        self._floating_point_type = floating_point_type

    @staticmethod
    def adjust_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2

    def train_epoch(self, train_loader, model, loss_function,
                    optimizer, epoch, writer_train, writer_val, val_loader):
        """
            Method that trains the model for one epoch on the training set and
            reports losses to Tensorboard using the writer_train

            train_loader: A DataLoader that contains the shuffled training set
            model: The model we are training
            loss_function: Loss function object
            optimizer: The optimizer we are using
            Epoch: integer epoch number
            writer_train: A Tensorboard summary writer for reporting the average
                loss while training.
            writer_val: A Tensorboard summary writer for reporting the average
                loss during validation.
            val_loader: A DataLoader that contains the shuffled validation set

        """
        losses = AverageMeter()
        model.train()

        if self._floating_point_type == "double":
            model = model.double()

        if hasattr(model, 'is_sequential'):
            is_sequential = True
        else:
            is_sequential = False

        timings_start = time.time()
        for i, data in enumerate(train_loader):
            if i % 20 == 0:
                if i > 0 and i % 100 == 0:
                    print("Elapsed time: {}".format(
                        str(time.time()-timings_start)))
                    timings_start = time.time()

                if is_sequential:
                    model.reset_state(batch=data['image'].shape[0],
                                      device=self._device)

                self.validate(self._device, val_loader, model,
                              loss_function, writer_val, i, epoch,
                              len(train_loader), 0.1)

                # switch to train mode
                model.train()

            if self._floating_point_type == "double":
                target_var = data["pose"].double().to(self._device)
                input_var = data["image"].double().to(self._device)
            else:
                target_var = data["pose"].float().to(self._device)
                input_var = data["image"].float().to(self._device)

            if torch.sum(torch.isnan(target_var)) > 0:
                continue

                # compute output
            if is_sequential:
                model.reset_state(batch=data['image'].shape[0],
                                  device=self._device)
                model.to(self._device)
            output = model(input_var)
            if loss_function.__class__.__name__ == "MSELoss":
                # norm over the last dimension (i.e. orientations)
                norms \
                    = torch.norm(output, dim=-1, keepdim=True).to(self._device)
                output = output / norms
            
            if loss_function.__class__.__name__ == "BinghamMixtureLoss":
                loss, log_likelihood = loss_function(target_var, output, epoch)

            else:
                loss, log_likelihood = loss_function(target_var, output)
            # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self._floating_point_type == "double":
                loss = loss.double() / data["image"].shape[0]
            else:
                loss = loss.float() / data["image"].shape[0]

            losses.update(loss.item(), data["image"].size(0))
            if i + len(train_loader) * epoch % 1000 == 0:
                Trainer.adjust_learning_rate(optimizer)
            writer_train.add_scalar('data/loss', loss,
                                    i + len(train_loader) * epoch)
            writer_train.add_scalar('data/log_likelihood', log_likelihood,
                                    i + len(train_loader) * epoch)
            cur_iter = epoch * len(train_loader) + i
            stats = loss_function.statistics(target_var, output, epoch)
            Trainer.report_stats(writer_train, stats, cur_iter)

            print("Epoch: [{0}][{1}/{2}]\t Loss {loss.last_val:.4f} "
                  "({loss.avg:.4f})\t".format(
                    epoch, i, len(train_loader), loss=losses))

    def validate(self, device, val_loader, model,  loss_function, writer,
                 index=None, cur_epoch=None, epoch_length=None, eval_fraction=1):
        """

        Method that validates the model on the validation set and reports losses
        to Tensorboard using the writer

        device: A string that states whether we are using GPU ("cuda:0") or cpu
        model: The model we are training
        loss_function: Loss function object
        optimizer: The optimizer we are using
        writer: A Tensorboard summary writer for reporting the average loss
            during validation.
        cur_epoch: integer epoch number representing the training epoch we are
            currently on.
        index: Refers to the batch number we are on within the training set
        epoch_length: The number of batches in an epoch
        val_loader: A DataLoader that contains the shuffled validation set
        loss_parameters: Parameters passed on to the loss generation class.

        """
        # switch to evaluate mode
        model.eval()

        losses = AverageMeter()
        log_likelihoods = AverageMeter()
        maads = AverageMeter()
        averaged_stats = AverageMeter()
        val_load_iter = iter(val_loader)
        
        for i in range(int(len(val_loader) * eval_fraction)):
            data = val_load_iter.next()

            if self._floating_point_type == "double":
                target_var = data["pose"].double().to(device)
                input_var = data["image"].double().to(device)
            else:
                target_var = data["pose"].float().to(device)
                input_var = data["image"].float().to(device)

            if torch.sum(torch.isnan(target_var)) > 0:
                continue
                # compute output
            output = model(input_var)
            if loss_function.__class__.__name__ == "MSELoss":
                # norm over the last dimension (ie. orientations)
                norms = torch.norm(output, dim=-1, keepdim=True).to(device)
                output = output / norms
            if loss_function.__class__.__name__ == "BinghamMixtureLoss":
                loss, log_likelihood = loss_function(target_var, output, cur_epoch)
            else:
                loss, log_likelihood = loss_function(target_var, output)

            if self._floating_point_type == "double":
                loss = loss.double() / data["image"].shape[0]
            else:
                loss = loss.float() / data["image"].shape[0]

            # measure accuracy and record loss
            losses.update(loss.item(), data["image"].size(0))
            log_likelihoods.update(log_likelihood.item(), data["image"].size(0))

            # TODO: Unify reporting to the style below.
            stats = loss_function.statistics(target_var, output, cur_epoch)
            averaged_stats.update(stats, data["image"].size(0))
   

        if index is not None:
            cur_iter = cur_epoch * epoch_length + index
            writer.add_scalar('data/loss', losses.avg, cur_iter)
            writer.add_scalar('data/log_likelihood', log_likelihoods.avg,
                              cur_iter)

            Trainer.report_stats(writer, averaged_stats.avg, cur_iter)

            print('Test:[{0}][{1}/{2}]\tLoss {loss.last_val:.4f} '
                  '({loss.avg:.4f})\t'.format(
                    cur_epoch, index, epoch_length, loss=losses))

    @staticmethod
    def report_stats(writer, stats, cur_iter):
        for key in stats:
            writer.add_scalar(
                'data/' + key, stats[key], cur_iter)


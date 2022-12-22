from datetime import datetime
import time
import os
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from typing import Iterable, Callable, Dict, Tuple


# AverageMeter class (taken from benatools: https://github.com/benayas1/benatools)
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Attributes
    ----------
    val : float
        Stores the average loss of the last batch
    avg : float
        Average loss
    sum : float
        Sum of all losses
    count : int
        number of elements
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates current internal state
        Parameters
        ----------
        val : float
            loss on each training step
        n : int, Optional
            batch size
        """
        if np.isnan(val) or np.isinf(val):
            return

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Fitter base class (taken from benatools: https://github.com/benayas1/benatools)
class DistributedTorchFitterBase:
    """
    Helper class to implement a training loop in PyTorch.
    PR (20/12/2022) update: Integration with Accelerate library.
    """
    def __init__(self,
                 model: torch.nn.Module = None,
                 loss: torch.nn.Module = None,
                 optimizer: torch.optim = None,
                 scheduler: torch.optim.lr_scheduler = None,
                 validation_scheduler: bool = True,
                 step_scheduler: bool = False,
                 folder: str = 'models',
                 verbose: bool = True,
                 save_log: bool = True,
                 world_size: int = 1, 
                 mixed_precision: str = 'no',
                 gradient_accumulation_steps: int = 1,
                 clip_value: float = 1.
                 ):
        """
        Args:
            model (torch.nn.Module): Model to be fitted
            loss (torch.nn.Module): DataFrame to split
            optimizer (torch.optim): Optimizer object
            scheduler (torch.optim.lr_scheduler, optional): Scheduler object. Defaults to None.
            validation_scheduler (bool, optional): Run scheduler step on the validation step. Defaults to True.
            step_scheduler (bool, optional): Run scheduler step on every training step. Defaults to False.
            folder (str, optional): Folder where to store checkpoints. Defaults to 'models'.
            verbose (bool, optional): Whether to print outputs or not. Defaults to True.
            save_log (bool, optional): Whether to write the log in log.txt or not. Defaults to True.
            world_size (int, optional): Number of cores running the processes. Defaults to 1.
            mixed_precision (str, optional): Mixed precision format to use (if any). 'fp16' requires python>=3.6,
                                             and 'bf16' requires python>=3.10.
            gradient_accumulation_steps (int, optional): Frequency of gradient update through batches. Defaults to 1.
            clip_value (float, optional): Range of gradient values. Defaults to 1.
        """

        # Miscellaneous
        if loss is not None:
            if type(loss) == type:
                self.loss_function = loss()
            else:
                self.loss_function = loss
        else:
            self.loss_function = None
        self.epoch = 0  # current epoch
        self.verbose = verbose
        self.base_dir = f'{folder}'
        self.logger = get_logger(__name__, log_level="DEBUG")
        self.save_log = save_log
        self.log_path = f'{self.base_dir}/log.txt'
        self.validation_scheduler = validation_scheduler
        self.step_scheduler = step_scheduler
        self.best_metric = 0
        self.world_size = world_size
        self.clip_value = clip_value

        # Accelerator object
        assert mixed_precision in ['no', 'fp16', 'bf16'], f"Mixed precision variable does not have a correct value (either 'no', 'fp16' or 'bf16')."
        self.scaler =  Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=gradient_accumulation_steps)

        # Model, optimizer and scheduler
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler


    def unpack(self, data):
        raise NotImplementedError('This class is a base class')

    def fit(self,
            train_dtl: torch.utils.data.DataLoader,
            val_dtl: torch.utils.data.DataLoader = None,
            n_epochs: int = 1,
            metrics: Iterable[Tuple[Callable[[Iterable, Iterable], float], dict]] = None,
            early_stopping: int = 0,
            early_stopping_mode: str = 'min',
            early_stopping_alpha: float = 0.0,
            early_stopping_pct: float = 0.0,
            save_checkpoint: bool = False,
            save_best_checkpoint: bool = True,
            verbose_steps: int = 0,
            step_callbacks: Iterable[Callable[[Dict], None]] = None,
            validation_callbacks: Iterable[Callable[[Dict], None]] = None):
        """
        Fits a model
        Args:
            train_dtl (torch.utils.data.DataLoader): Training data
            val_dtl (torch.utils.data.DataLoader, optional): Validation Data. Defaults to None.
            n_epochs (int, optional): Maximum number of epochs to train. Defaults to 1.
            metrics ( function with (y_true, y_pred, **metric_kwargs) signature, optional): Metric to evaluate results on. Defaults to None.
            metric_kwargs (dict, optional): Arguments for the passed metric. Ignored if metric is None. Defaults to {}.
            early_stopping (int, optional): Early stopping epochs. Defaults to 0.
            early_stopping_mode (str, optional): Min or max criteria. Defaults to 'min'.
            early_stopping_alpha (float, optional): Value that indicates how much to improve to consider early stopping. Defaults to 0.0.
            early_stopping_pct (float, optional): Value between 0 and 1 that indicates how much to improve to consider early stopping. Defaults to 0.0.
            save_checkpoint (bool, optional): Whether to save the checkpoint when training. Defaults to False.
            save_best_checkpoint (bool, optional): Whether to save the best checkpoint when training. Defaults to True.
            verbose_steps (int, optional): Number of step to print every training summary. Defaults to 0.
            step_callbacks (list of callable, optional): List of callback functions to be called after logging steps
            validation_callbacks (list of callable, optional): List of callback functions to be called after an epoch
        Returns:
            pd.DataFrame: DataFrame containing training history
        """
        if self.model is None or self.loss_function is None or self.optimizer is None:
            self.log(f"ERROR: Either model, loss function or optimizer is not existing.")
            raise ValueError(f"ERROR: Either model, loss function or optimizer is not existing.")

        if self.best_metric == 0.0:
            self.best_metric = np.inf if early_stopping_mode == 'min' else -np.inf

        initial_epochs = self.epoch

        # Use the same train loader for validation. A possible use case is for autoencoders
        if isinstance(val_dtl, str) and val_dtl == 'training':
            val_dtl = train_dtl
        
        # Save batch size (counting number of cores already)
        self.batch_size = self.world_size*train_dtl.batch_size

        # Prepare objects
        train_dtl, val_dtl, self.model, self.optimizer, self.scheduler = self.scaler.prepare(
            train_dtl,
            val_dtl,
            self.model,
            self.optimizer,
            self.scheduler
        )

        # Training/validation loop
        training_history = []
        es_epochs = 0
        for e in range(n_epochs):
            history = {'epoch': e}  # training history log for this epoch

            # Update log
            lr = self.optimizer.param_groups[0]['lr']
            self.log(f'\n{datetime.utcnow().isoformat(" ", timespec="seconds")}\n \
                        EPOCH {str(self.epoch+1)}/{str(n_epochs+initial_epochs)} - LR: {lr}')

            # Run one training epoch
            t = time.time()
            train_summary_loss = self.train_one_epoch(train_dtl, step_callbacks=step_callbacks, verbose_steps=verbose_steps)
            history['train_loss'] = train_summary_loss.avg  # training loss
            history['lr'] = self.optimizer.param_groups[0]['lr']

            # Save checkpoint
            if save_checkpoint:
                self.save(f'{self.base_dir}/last-checkpoint.bin', False)

            if val_dtl is not None:
                # Run epoch validation
                val_summary_loss, calculated_metrics = self.validation(val_dtl,
                                                                       metric=metrics,
                                                                       verbose_steps=verbose_steps)
                history['val_loss'] = val_summary_loss.avg  # validation loss

                # Write log
                metric_log = ' - ' + ' - '.join([f'{fname}: {value}' for value, fname in calculated_metrics]) if calculated_metrics else ''
                self.log(f'\r[RESULT] {(time.time() - t):.2f}s - train loss: {train_summary_loss.avg:.5f} - val loss: {val_summary_loss.avg:.5f}' + metric_log)

                if calculated_metrics:
                    history.update({fname: value for value, fname in calculated_metrics})
                    #history['val_metric'] = calculated_metrics

                calculated_metric = calculated_metrics[0][0] if calculated_metrics else val_summary_loss.avg
            else:
                # If no validation is provided, training loss is used as metric
                calculated_metric = train_summary_loss.avg

            es_pct = early_stopping_pct * self.best_metric

            # Check if result is improved, then save model
            if (
                ((metrics) and
                 (
                  ((early_stopping_mode == 'max') and (calculated_metric - max(early_stopping_alpha, es_pct) > self.best_metric)) or
                  ((early_stopping_mode == 'min') and (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric))
                 )
                ) or
                ((metrics is None) and
                 (calculated_metric + max(early_stopping_alpha, es_pct) < self.best_metric) # the standard case is to minimize
                )
               ):
                self.log(f'Validation metric improved from {self.best_metric} to {calculated_metric}')
                self.best_metric = calculated_metric
                self.model.eval()
                if save_best_checkpoint:
                    savepath = f'{self.base_dir}/best-checkpoint.bin'
                    self.save(savepath)
                es_epochs = 0  # reset early stopping count
            else:
                es_epochs += 1  # increase epoch count with no improvement, for early stopping check

            # Callbacks receive the history dict of this epoch
            if validation_callbacks is not None:
                if not isinstance(validation_callbacks, list):
                    validation_callbacks = [validation_callbacks]
                for c in validation_callbacks:
                    c(history, step=(self.epoch+1)*len(train_dtl))

            # Check if Early Stopping condition is met
            if (early_stopping > 0) & (es_epochs >= early_stopping):
                self.log(f'Early Stopping: {early_stopping} epochs with no improvement')
                training_history.append(history)
                break

            # Scheduler step after validation
            if self.validation_scheduler and self.scheduler is not None:
                self.scheduler.step(metrics=calculated_metric)

            training_history.append(history)
            self.epoch += 1

        return pd.DataFrame(training_history).set_index('epoch')

    def train_one_epoch(self, train_dtl, step_callbacks=None, verbose_steps=0):
        """
        Run one epoch on the train dataset
        Parameters
        ----------
        train_dtl : torch.data.utils.DataLoader
            DataLoaders containing the training dataset
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        """
        self.model.train()  # set train mode
        summary_loss = AverageMeter()  # object to track the average loss
        t = time.time()

        # run epoch
        for step, data in enumerate(train_dtl):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rTrain Step {step}/{len(train_dtl)} | ' +
                        f'train_loss: {summary_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs | ' +
                        f'ETA: {(len(train_dtl)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
                    # Callbacks receive the history dict of this epoch
                    if (step_callbacks is not None) & (step>0):
                        if not isinstance(step_callbacks, list):
                            step_callbacks = [step_callbacks]
                        for c in step_callbacks:
                            c({'train_loss':summary_loss.avg,
                               'lr':self.optimizer.param_groups[0]['lr']}, step=self.epoch*len(train_dtl)+step)
            # Unpack batch of data
            x, y = self.unpack(data)

            # Run one batch
            loss = self.train_one_batch(x, y)

            summary_loss.update(loss.detach().item(), self.batch_size)

            # Update optimizer
            self.optimizer.step()

            # LR Scheduler step after epoch
            if self.step_scheduler and self.scheduler is not None:
                self.scheduler.step()   

        self.log(f'\r[TRAIN] {(time.time() - t):.2f}s - train loss: {summary_loss.avg:.5f}')

        return summary_loss

    def train_one_batch(self, x, y):
        """
        Trains one batch of data.
        The actions to be done here are:
        - extract x and y (labels)
        - calculate output and loss
        - backpropagate
        Args:
            x (List or Tuple or Dict): Data
            y (torch.Tensor): Labels
            w (torch.Tensor, optional): Weights. Defaults to None.
        Returns:
            torch.Tensor: A tensor with the calculated loss
        """
        with self.scaler.accumulate(self.model):
            # Reset gradients
            self.optimizer.zero_grad()
            # Get logits
            if isinstance(x, tuple) or isinstance(x, list):
                output = self.model(*x)
            elif isinstance(x, dict):
                output = self.model(**x)
            else:
                output = self.model(x)
            # Compute loss
            loss = self.loss_function(output, y)
            # Reduce loss (weights are left to custom loss implementation)
            loss = self.scaler.reduce(loss, reduction='sum')
            # Backpropagation
            self.scaler.backward(loss)
            if self.scaler.sync_gradients:
                self.scaler.clip_grad_value_(self.model.parameters(), self.clip_value)
        return loss

    def validation(self, val_dtl, metric=None, verbose_steps=0):
        """
        Validates a model
        Parameters
        ----------
        val_dtl : torch.utils.data.DataLoader
            Validation Data
        metric : function with (y_true, y_pred, **metric_kwargs) signature
            Metric to evaluate results on
        metric_kwargs : dict
            Arguments for the passed metric. Ignored if metric is None
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        AverageMeter
            Object with this epochs's average loss
        float
            Calculated metric if a metric is provided, else None
        """
        if self.model is None or self.loss_function is None or self.optimizer is None:
            self.log(f"ERROR: Either model, loss function or optimizer is not existing.")
            raise ValueError(f"ERROR: Either model, loss function or optimizer is not existing.")

        self.model.eval()
        summary_loss = AverageMeter()
        y_preds = []
        y_true = []

        t = time.time()
        for step, data in enumerate(val_dtl):
            if self.verbose & (verbose_steps > 0):
                if step % verbose_steps == 0:
                    print(
                        f'\rVal Step {step}/{len(val_dtl)} | ' +
                        f'val_loss: {summary_loss.avg:.5f} | ' +
                        f'time: {(time.time() - t):.2f} secs |' +
                        f'ETA: {(len(val_dtl)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():
                x, y = self.unpack(data)
                if metric:
                    y_true += y.cpu().numpy().tolist()
                # Get model logits
                if isinstance(x, tuple) or isinstance(x, list):
                    output = self.model(*x)
                elif isinstance(x, dict):
                    output = self.model(**x)
                else:
                    output = self.model(x)
                # Compute loss
                loss = self.loss_function(output, y)
                # Reduce loss (weights are left to custom loss implementation)
                loss = self.scaler.reduce(loss, reduction='sum')
                # Update metrics tracker objects
                summary_loss.update(loss.detach().item(), self.batch_size)
                # Gather tensors for metric calculation
                if metric:
                    output = self.scaler.gather(output)
                    y_preds += output.cpu().numpy().tolist()

        # Callback metrics
        metric_log = ' '*30
        if metric:
            calculated_metrics = []
            y_pred = np.argmax(y_preds, axis=1)
            for f, args in metric:
                value = f(y_true, y_pred, **args)
                calculated_metrics.append((value, f.__name__))
                metric_log = f'- {f.__name__} {value:.5f} '
        else:
            calculated_metrics = None

        self.log(f'\r[VALIDATION] {(time.time() - t):.2f}s - val. loss: {summary_loss.avg:.5f} ' + metric_log)
        return summary_loss, calculated_metrics

    def predict(self, test_loader, verbose_steps=0):
        """
        Makes predictions using the trained model
        Parameters
        ----------
        test_loader : torch.utils.data.DataLoader
            Test Data
        verbose_steps : int, defaults to 0
            number of step to print every training summary
        Returns
        -------
        np.array
            Predicted values by the model
        """
        if self.model is None:
            self.log(f"ERROR: Model is not existing.")
            raise ValueError(f"ERROR: Model is not existing.")

        self.model.eval()
        y_preds = []
        t = time.time()

        for step, data in enumerate(test_loader):
            if self.verbose & (verbose_steps > 0) > 0:
                if step % verbose_steps == 0:
                    print(
                        f'\rPrediction Step {step}/{len(test_loader)} | ' +
                        f'time: {(time.time() - t):.2f} secs |' +
                        f'ETA: {(len(test_loader)-step)*(time.time() - t)/(step+1):.2f}', end=''
                    )
            with torch.no_grad():  # no gradient update
                x, _, _ = self.unpack(data)

                # Output
                if isinstance(x, tuple) or isinstance(x, list):
                    output = self.model(*x)
                elif isinstance(x, dict):
                    output = self.model(**x)
                else:
                    output = self.model(x)
                output = self.scaler.gather(output)
                y_preds += output.cpu().numpy().tolist()

        return np.array(y_preds)

    def save(self, path, verbose=True):
        """
        Save model and other metadata
        Args:
            path (str): Path of the file to be saved
            verbose (bool, optional): True = print logs, False = silence. Defaults to True.
        """
        # Fetch model without distributed config
        model_aux = self.scaler.unwrap_model(self.model)

        if verbose:
            self.log(f'Checkpoint is saved to {path}')
        model_aux.eval()

        data = {
                'model_state_dict': model_aux.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_summary_loss': self.best_metric,
                'epoch': self.epoch,
        }

        if self.scheduler is not None:
            data['scheduler_state_dict'] = self.scheduler.state_dict()

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.scaler.save(data, path)

    def load(self, path, only_model=False):
        """
        Load model and other metadata
        Args:
            path (str): Path of the file to be loaded
            only_model (bool, optional): Whether to load just the model weights. Defaults to False.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if only_model:
            return

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_metric = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    @staticmethod
    def load_model_weights(path, model):
        """
        Static method that loads weights into a torch module, extracted from a checkpoint
        Args:
            path (str): Path containing the weights. Normally a .bin or .tar file
            model (torch.nn.Module): Module to load the weights on
        Returns:
            torch.nn.Module: The input model with loaded weights
        """

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def log(self, message):
        """
        Log training ouput into console and file
        Args:
            message (str): Message to be logged
        """
        if self.verbose:
            self.logger.debug(message, main_process_only=True)

        if self.save_log is True:
            if not os.path.exists(self.base_dir):
                os.makedirs(self.base_dir)
            with open(self.log_path, 'a+') as logger:
                logger.write(f'{message}\n')


# Customised class
class SegmFitter(DistributedTorchFitterBase):
    def unpack(self, data):
        return data
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer


class BaseScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        # if not isinstance(optimizer, Optimizer):
        #     raise TypeError('{} is not an Optimizer'.format(
        #         type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr * param_group['lr_factor']


class CosineWithRestarts(BaseScheduler):  # pylint: disable=protected-access
    """
    Cosine annealing with restarts.
    This is decribed in the paper https://arxiv.org/abs/1608.03983.
    Parameters
    ----------
    optimizer : ``torch.optim.Optimizer``
    cycle_len : ``int``
        The maximum number of iterations within the first cycle.
    lr_min : ``float``, optional (default=0)
        The minimum learning rate.
    last_epoch : ``int``, optional (default=-1)
        The index of the last epoch. This is used when restarting.
    factor : ``float``, optional (default=1)
        The factor by which the cycle length (``T_max``) increases after each restart.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cycle_len: int,
                 lr_min: float = 0.0,
                 last_epoch: int = -1,
                 factor: float = 1.0,
                 gamma: float = 1.0) -> None:
        assert cycle_len > 0
        assert lr_min >= 0
        if cycle_len == 1 and factor == 1:
            print("Cosine annealing scheduler will have no effect on the learning "
                  "rate since T_max = 1 and factor = 1.")
        self.cycle_len = cycle_len
        self.lr_min = lr_min
        self.factor = factor
        self.gamma = gamma
        self._last_restart = 0
        self._cycle_counter = 0
        self._cycle_factor = 1.
        self._updated_cycle_len = cycle_len
        self._initialized = False
        self.coef = 1.0
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time ``self.get_lr()`` was called,
        # since ``torch.optim.lr_scheduler._LRScheduler`` will call ``self.get_lr()``
        # when first initialized, but the learning rate should remain unchanged
        # for the first epoch.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            self.lr_min + ((self.coef * lr - self.lr_min) / 2) * (
                        np.cos(
                                np.pi *
                                (self._cycle_counter % self._updated_cycle_len) /
                                self._updated_cycle_len
                        ) + 1
                )
                for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self.coef *= self.gamma
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.cycle_len)
            self._last_restart = step

        return lrs

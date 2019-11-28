import torch
import torch.nn as nn
from collections import OrderedDict
from catalyst.dl import SupervisedRunner
from apex import amp
from catalyst.contrib.criterion import HuberLoss, MSELoss, L1Loss
from catalyst.contrib.optimizers import RAdam, Lookahead

from dataset import get_dataloaders

from catalyst.dl.callbacks import DiceCallback, IouCallback, \
  CriterionCallback, CriterionAggregatorCallback, SchedulerCallback

from model import Forecaster, CellForecaster
from schedulers import CosineWithRestarts
from metrics import MAEMetric
from utils import set_seed

import os
os.environ['OMP_NUM_THREADS'] = '1'

BATCH_SIZE = 32
SEQ_LEN = 32
FUTURE = 32
LR_MAX = 0.001
EPOCHS = 8


def train():
    set_seed(42)

    loaders = OrderedDict()
    train_loader, valid_loader = get_dataloaders(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, future=FUTURE)
    loaders['train'] = train_loader
    loaders['valid'] = valid_loader

    # logdir = './logs/lstm_baseline'  # logs/lstm_baseline/checkpoints/train.255.pth    38.9158
    # logdir = './logs/lstm_baseline_hidden_256'  # logs/lstm_baseline_hidden_256/checkpoints/train.240.pth 10.7952
    # logdir = './logs/lstm_baseline_hidden_1024'  # logs/lstm_baseline_hidden_1024/checkpoints/train.150.pth 7.3388
    # logdir = './logs/lstm_baseline_hidden_1024_cyclic'  #
    # logs/lstm_baseline_hidden_1024_cyclic/checkpoints/train.240.pth 7.7115
    # logdir = './logs/lstm_baseline_hidden_1024_cyclic_lr_0.004'  # logs/lstm_baseline_hidden_1024_cyclic_lr_0.004/
    # checkpoints/train.150.pth 7.0322
    # logdir = './logs/lstm_cell_1024_lr_0.001'  #
    # logdir = './logs/lstm_cell_1024_lr_0.001_seq_len_8_future_12'  # logs/lstm_cell_1024_lr_0.001_seq_len_8_future_12
    # /checkpoints/train.110.pth      94.3721
    # logdir = './logs/lstm_cell_1024_lr_0.001_seq_len_6_future_12'  #
    # logdir = './logs/lstm_cell_1024_lr_0.001_seq_len_8_future_8'  # logs/lstm_cell_1024_lr_0.001_seq_len_8_future_8/
    # checkpoints/train.102.pth       70.2596
    # logdir = './logs/lstm_cell_2048_lr_0.001_seq_len_8_future_8'  # logs/lstm_cell_2048_lr_0.001_seq_len_8_future_8/
    # checkpoints/train.69.pth 63.9701
    # logdir = './logs/lstm_cell_2048_lr_0.001_seq_len_8_future_8_start_at_center'
    # logdir = './logs/lstm_cell_2048_lr_0.001_seq_len_12_future_4_velocity'
    # logdir = './logs/MOT_data_lstm_cell_1024_lr_0.001_seq_len_16_future_16'
    # logdir = './logs/MOT_data_lstm_cell_1024_lr_0.001_seq_len_16_future_16_drop_0.5'
    # logdir = './logs/MOT_data_lstm_cell_1024_lr_0.001_seq_len_16_future_16_drop_0.5_flips'
    # logdir = './logs/MOT_data_lstm_cell_1024_lr_0.001_seq_len_16_future_16_flips_shift'
    # logdir = './logs/MOT_data_lstm_cell_1024_lr_0.001_seq_len_32_future_16_flips_shift'
    # logdir = './logs/MOT_data_lstm_cell_1024_lr_0.001_seq_len_32_future_32_drop_0.2_flips_shift'
    # logdir = './logs/MOT_data_lstm_cell_1024_lr_0.001_seq_len_32_future_32_drop_0.2_flips_shift_all_seq_loss'

    # logdir = './logs/MOT_data_lstm_cell_1024_lr_0.001_seq_len_32_future_32_drop_0.2_flips_shift_all_seq_loss'
    # logdir = './logs/MOT_data_lstm_cell_1024_len_32_future_32_batch_128'
    # logdir = './logs/MOT_data_lstm_cell_1024_len_32_future_32_batch_64'
    # logdir = './logs/MOT_data_lstm_cell_1024_len_32_future_32_batch_32'
    # logdir = './logs/MOT_data_lstm_cell_1024_len_32_future_32_huber_200'
    # logdir = './logs/MOT_data_lstm_cell_1024_len_32_future_32_batch_16_huber_200'
    # logdir = './logs/MOT_data_lstm_cell_1024_len_32_future_32_batch_32_huber_1.0'
    # logdir = './logs/MOT_data_lstm_cell_1024_len_32_future_32_batch_32'
    logdir = './logs/MOT_data_lstm_cell_128_len_32_future_32_batch_32'

    # model = Forecaster(input_size=2, hidden_size=1024, output_size=2).cuda()

    model = CellForecaster(
        input_size=2,
        hidden_size=128,
        output_size=2,
        future=FUTURE
    ).cuda()

    criterion = {
        # 'huber': HuberLoss(),
        # 'huber': HuberLoss(clip_delta=1.0),
        'huber': L1Loss(),
        # 'huber': MSELoss(),
    }

    optimizer = RAdam([{'params': model.parameters(), 'lr_factor': 1.0}], lr=LR_MAX)

    scheduler = CosineWithRestarts(
        optimizer,
        cycle_len=32 * len(train_loader),
        lr_min=0.0004,
        factor=1.4,
        gamma=0.85
    )

    runner = SupervisedRunner(input_key='input', input_target_key='target')

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        # scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=EPOCHS,
        verbose=True,
        # fp16=True,
        callbacks=[
            CriterionCallback(
                input_key='target',
                prefix='loss_huber',
                criterion_key='huber'
            ),
            CriterionAggregatorCallback(
                prefix='loss',
                loss_keys=['loss_huber'],
                loss_aggregate_fn='sum'
            ),
            MAEMetric(
                input_key='target',
                output_key='logits',
                prefix='mae'
            )
            # SchedulerCallback(mode='batch')
        ]
    )


if __name__ == '__main__':
    train()

import os
import torch.nn as nn
from pathlib import Path
from Seismic_data_reconstruction.Model.AFFNet.down_unet import AFFNet
from Seismic_data_reconstruction.Train.utils import *
from Seismic_data_reconstruction.Train.DataLoader import get_AllDataLoader
from Seismic_data_reconstruction.Losses.Self_supervised_MAE_MSE_loss import *

class TrainConfig:
    file_name = str(Path(os.path.basename(__file__)).stem)
    Base_path = str(Path(os.path.abspath(__file__)).parent.parent.parent)

    data_path = os.path.join(Base_path, 'Data', 'Mobil_AVO_viking_graben_line_12', 'AVO_clean.npy')
    mask_path = os.path.join(Base_path, 'Data', 'Mobil_AVO_viking_graben_line_12', 'AVO_mask_1_95.npy')
    uniform_path = None
    fkmask_path = None
    get_AllDataLoader = get_AllDataLoader

    device = get_Device()

    Model = AFFNet
    blindspot = True
    dataset = 'MAVGL12'
    method = 'AFFNet'
    in_channels = 1
    out_channels = 1
    zero_output_weights = False
    batch_size = 1

    Loss = MAE_loss
    MetricsLogger = SNRLogger

    OptimPipeLineConfig = pipeline
    clip_grad = False
    clip_grad_value = 1

    pretrain = None
    resume = False
    load_best = False

    save_path = os.path.join(Base_path, 'exp', Model.__name__, file_name)
    exp_dir = save_path

    optim = torch.optim.Adam
    min_lr = 1e-5
    lr = 1e-3
    warmup_epochs = 10
    num_epoches = 100
    test_interval = 10

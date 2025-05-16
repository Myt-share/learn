import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from IEDM import Solver
from utils1 import *
import random
import os
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TKAgg')
# 统一随机数种子
init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


class SeismicData(Dataset):
    def __init__(self,
                 data_dir,
                 mask_dir):
        self.data = np.load(data_dir)
        self.mask = np.load(mask_dir)


    def __len__(self):
        return len(self.data)


    def __getitem__(self,index):
        c = self.data[index]
        m = self.mask[index]
        c = self.log_preprocessing(c)

        return c.dot(m), m, c

    def log_preprocessing(self, x, s=1):
        return np.sign(x) * np.log10(np.abs(s * x) + 1)


def test_data(args):
    data_dir = args.data_dir
    mask_dir = args.mask_dir
    batch_size = args.batch_size


    dset = SeismicData(data_dir, mask_dir)
    data_loader = DataLoader(dset,
                             batch_size=batch_size,
                             shuffle=False)
    return data_loader


########################################################################################################################################################################################################
def train_arg():
    parser = argparse.ArgumentParser(description='work')
    parser.add_argument('--cuda', default=True, type=bool, help='enable cuda')
    parser.add_argument('--dataset', default='MAVGL12', type=str, help='dataset name')  # MAVGL12
    parser.add_argument('--data_dir', default=r'E:/muyuting/data/AVOclean.npy', type=str,
                        help='dataset directory')
    parser.add_argument('--mask_dir', default=r'E:/muyuting/data/mask_AVO_30_1.npy', type=str,
                        help='dataset directory')
    # parser.add_argument('--dataset', default='M94', type=str, help='dataset name')
    # parser.add_argument('--data_dir', default=r'E:/muyuting/data/M94_clean.npy', type=str,
    #                     help='dataset directory')  # M94_clean,M94_mask_1_75
    # parser.add_argument('--mask_dir', default=r'E:/muyuting/data/M94_mask_1_75.npy', type=str, help='dataset directory')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate of the model')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the model')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='beta2 parameter of the Adam optimizer for the model')
    parser.add_argument('--max_iter', default=200, type=float, help='maximum training iteration')
    parser.add_argument('--print_iter', default=1, type=int, help='print losses iter')
    # parser.add_argument('--obratio', default=0.5, type=float, help='-')

    # Checkpoint
    parser.add_argument('--ckpt_dir', default=r'E:/muyuting/EDMLearing/checkpoint', type=str,
                        help='checkpoint directory')#
    parser.add_argument('--ckpt_save_iter', default=1, type=float, help='checkpoint save iter')
    parser.add_argument('--save_image', default=r'E:/muyuting/EDMLearing/reconstruct', type=str,
                        help='reconstruct data')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')

    return parser.parse_args()


def test_arg():
    parser = argparse.ArgumentParser(description='work')
    parser.add_argument('--cuda', default=True, type=bool, help='enable cuda')
    parser.add_argument('--dataset', default='MAVGL12', type=str, help='dataset name')
    parser.add_argument('--data_dir', default=r'E:/muyuting/data/AVOclean.npy', type=str,
                        help='dataset directory')
    parser.add_argument('--mask_dir', default=r'E:/muyuting/data/mask_AVO_30_1.npy', type=str,
                        help='dataset directory')
    # parser.add_argument('--dataset', default='M94', type=str, help='dataset name')
    # parser.add_argument('--data_dir', default=r'E:/muyuting/data/M94_clean.npy', type=str,
    #                     help='dataset directory')  # M94_clean,M94_mask_1_75
    # parser.add_argument('--mask_dir', default=r'E:/muyuting/data/M94_mask_1_75.npy', type=str, help='dataset directory')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    return parser.parse_args()


class test(object):
    def __init__(self):
        self.solver = Solver(train_arg())
        self.model = self.solver.model
        self.device = self.solver.device
        self.data_loader = test_data(test_arg())

    def call(self, n):
        self.load_checkpoint(n)
        MAE = 0
        SNR = 0
        num = 0

        for i, (x, m, y) in enumerate(self.data_loader):
            num += x.shape[0]
            x = x.float()
            x = torch.FloatTensor(x).unsqueeze(1).to(self.device)
            m = torch.FloatTensor(m).unsqueeze(1).to(self.device)
            y = torch.FloatTensor(y).unsqueeze(1).to(self.device)
            e = torch.eye(x.size(-1)).expand((x.size(0), 1, x.size(-1), x.size(-1))).to(self.device)

            with torch.no_grad():
                r = self.model(x)
                r = r.matmul(e - m) + x

                plt.figure()
                plt.imshow(r[0, 0, :, :].cpu().numpy(), cmap='gray',aspect=0.2)
                plt.show()
                break



            MAE += (r - y).abs().mean([1, 2, 3]).sum()
            SNR += self.snr(r, y)
        print('epoch:{:.1f} mae:{:.7f} snr:{:.7f}'.format(n, MAE.item() / num, SNR.item() / num))

    def snr(self, x, y):
        n = torch.Tensor([20]).to(self.device)
        return (n * torch.log10(y.square().sum([1, 2, 3]).sqrt() / (x - y).square().sum([1, 2, 3]).sqrt())).sum()
    def load_checkpoint(self, i):
        # filepath = r'D:/gengxin/S2S/S2SR/checkpoint/' + str(i)
        filepath = r'E:/muyuting/新参数/MABTN/AVO_75/' + str(i)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
        self.model.load_state_dict(checkpoint['model_states']['model'])
        # self.model.load_state_dict(checkpoint['model_states']['model'])

if __name__ == "__main__":
    T = test()
    for i in range(62, 63):
        # if i == 0:
        #     pass
        # else:
        #     T.solver.load_checkpoint(str(i * 1))
        T.call(i*1)

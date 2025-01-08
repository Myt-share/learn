import torch
import torch.fft
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random


class SeismicData(Dataset):
    def __init__(self, data_dir, mask_dir):
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
        # return x / abs(x).max(axis=0)


def return_data(args):
    data_dir = args.data_dir
    batch_size = args.batch_size
    mask_dir = args.mask_dir

    dset = SeismicData(data_dir, mask_dir)
    data_loader = DataLoader(dset,
                             batch_size=batch_size,
                             shuffle=True)
    return data_loader


def mask_sampling(m, rato=0.05, missing_type='1'):
    # m: [B,H,W]
    # rato: non-missing rate
    n_trace = m.shape[-1]
    n = int(n_trace * rato)
    M = np.zeros(m.shape)
    if missing_type == '1':
        e = np.ones(n_trace)
        for i in range(1):
            index, _ = np.where((e - m[i].detach().cpu().numpy()) == 0)

            samples = random.sample(range(0, len(index)), n)
            indexR = index[samples]
            M[i, indexR, indexR] = 1

        return torch.tensor(M, dtype=torch.float, requires_grad=False)
    elif missing_type == '0':
        for i in range(len(m)):
            e = np.ones(n_trace)
            index,_ = np.where((e - m[i].detach().cpu().numpy()) == 0)

            samples = random.sample(range(0, len(index)), n)
            a = index[samples]
            M[i, a, a] = 1
        return torch.tensor(M, dtype=torch.float, requires_grad=False)
    else:
        for i in range(len(m)):
            e = np.ones(n_trace)
            index, _ = np.where((e - m[i].detach().cpu().numpy()) == 0)
            d = int(n / 2)  # 缺失半径
            c = np.random.choice(index)  # 缺失迹线中点
            index = list(set(range(0, n_trace)) - set(range(c - d, c + d)))
            M[0, index, index] = 1
        return torch.tensor(M, dtype=torch.float, requires_grad=False)
def mask_sampling1(m, rato=0.05, missing_type='1'):
    # m: [B,H,W]
    # rato: non-missing rate
    n_trace = m.shape[-1]
    n = int(n_trace * rato)
    M = np.zeros(m.shape)
    if missing_type == '1':
        e = np.ones(n_trace)
        for i in range(1):
            index, _ = np.where((e - m[i].detach().cpu().numpy()) == 1)

            samples = random.sample(range(0, len(index)), n)
            indexR = index[samples]
            M[i, indexR, indexR] = 1

        return torch.tensor(M, dtype=torch.float, requires_grad=False)
    elif missing_type == '0':
        for i in range(len(m)):
            samples = random.sample(range(0, n_trace), n)
            M[i, samples, samples] = 1
        return torch.tensor(M, dtype=torch.float, requires_grad=False)
    else:
        for i in range(len(m)):
            d = int(n / 2)  # 缺失半径
            c = random.randint(d,n_trace-1-d)  # 缺失迹线中点
            index = list(set(range(0, n_trace)) - set(range(c - d, c + d)))
            M[0, index, index] = 1

        return torch.tensor(M, dtype=torch.float, requires_grad=False)



if __name__ == '__main__':

    x = np.load('E:/muyuting/data/AVOclean.npy')[0]

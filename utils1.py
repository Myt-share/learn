import torch
import torch.nn as nn
import torch.fft
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import random


class SeismicData(Dataset):
    def __init__(self,
                 data_dir,
                 mask_dir):#,fkmask_dir ):
        self.data = np.load(data_dir)
        self.mask = np.load(mask_dir)
        # self.fkmask = np.load(fkmask_dir)
        # self.mask1 = np.load(mask1_dir)


    def __len__(self):
        return len(self.data)


    def __getitem__(self,index):
        c = self.data[index]
        m = self.mask[index]
        c = self.log_preprocessing(c)
        # m1 = self.mask1[index]

        return c.dot(m), m, c

    def log_preprocessing(self, x, s=1):
        return np.sign(x) * np.log10(np.abs(s * x) + 1)
        # return x / abs(x).max(axis=0)


def return_data(args):
    data_dir = args.data_dir
    batch_size = args.batch_size
    mask_dir = args.mask_dir
    # fkmask_dir = args.fkmask_dir
    # mask1_dir = args.mask1_dir

    dset = SeismicData(data_dir, mask_dir)#, fkmask_dir)
    data_loader = DataLoader(dset,
                             batch_size=batch_size,
                             shuffle=True)
    return data_loader

class SeismicData1(Dataset):
    def __init__(self,
                     data_dir,
                     mask_dir, fkmask_dir):
        self.data = np.load(data_dir)
        self.mask = np.load(mask_dir)
        self.fkmask = np.load(fkmask_dir)
            # self.mask1 = np.load(mask1_dir)

    def __len__(self):
        return len(self.data)
    def select_index(self, index):
        # py = np.random.randint(-1, 2, 1)[0]
        if random.random() > 0.5:
            py = np.random.randint(-1, 0, 1)[0]
        else:
            py = np.random.randint(1, 2, 1)[0]
        return min(max(0, index + py), len(self.data)-1)

    def __getitem__(self,index):
        c = self.data[index]
        c = self.log_preprocessing(c, 1)#, 1e7
        # c = c + self.noise[index]
        cm = self.mask[index]
        if index == 0:
            index1 = np.random.randint(1, 2, 1)[0]
            # index1 = np.random.randint(0, 2, 1)[0]
        elif index == (len(self.data) - 1):
            index1 = np.random.randint(-1, 0, 1)[0]
            # index1 = np.random.randint(-1, 1, 1)[0]
        else:
            index1 = self.select_index(index)
        p = self.data[index1]
        p = self.log_preprocessing(p, 1)#, 1e7
        # p = p + self.noise[index1]
        pm = self.mask[index1]
        return c.dot(cm), p.dot(pm), cm, pm
    def log_preprocessing(self, x, s=1):
        return np.sign(x) * np.log10(np.abs(s * x) + 1)
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
            # index, _ = np.where((e - m[i].detach().cpu().numpy()) == 1)
            # index, _ = np.where((e - m) == 0)

            samples = random.sample(range(0, len(index)), n)
            indexR = index[samples]
            M[i, indexR, indexR] = 1

        return torch.tensor(M, dtype=torch.float, requires_grad=False)

        # return M
    elif missing_type == '0':
        for i in range(len(m)):
            e = np.ones(n_trace)
        # for i in range(1):
            index,_ = np.where((e - m[i].detach().cpu().numpy()) == 0)
            # print(index)
            # index = random.sample(range(1, int(n_trace / n)), 1)[0]
            # samples = list(range(index, n_trace, int(n_trace / n)))
            #
            samples = random.sample(range(0, len(index)), n)
            # print(samples)
            a = index[samples]
            # print(a)
            M[i, a, a] = 1
            # M[i, samples, samples] = 1
        return torch.tensor(M, dtype=torch.float, requires_grad=False)
    else:
        for i in range(len(m)):
            e = np.ones(n_trace)
            index, _ = np.where((e - m[i].detach().cpu().numpy()) == 0)
            d = int(n / 2)  # 缺失半径
            c = np.random.choice(index)  # 缺失迹线中点
            index = list(set(range(0, n_trace)) - set(range(c - d, c + d)))
            M[0, index, index] = 1
            # index = random.sample(range(1, int(n_trace / n)), 1)[0]
            # samples = list(range(index, n_trace, int(n_trace / n)))
            #
            # samples = random.sample(range(0, n_trace), n)
            # M[i, samples, samples] = 1
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
            # index, _ = np.where((e - m[i].detach().cpu().numpy()) == 0)
            index, _ = np.where((e - m[i].detach().cpu().numpy()) == 1)
            # index, _ = np.where((e - m) == 0)

            samples = random.sample(range(0, len(index)), n)
            indexR = index[samples]
            M[i, indexR, indexR] = 1

        return torch.tensor(M, dtype=torch.float, requires_grad=False)
    elif missing_type == '0':
        for i in range(len(m)):
        # e = np.ones(n_trace)
        # for i in range(1):
        #     index = np.where((e - m[i].detach().cpu().numpy()) == 0)
            # index = random.sample(range(1, int(n_trace / n)), 1)[0]
            # samples = list(range(index, n_trace, int(n_trace / n)))
            #
            samples = random.sample(range(0, n_trace), n)
            # a = index[samples]
            # M[i, a, a] = 1
            M[i, samples, samples] = 1
        return torch.tensor(M, dtype=torch.float, requires_grad=False)
    else:
        for i in range(len(m)):
            # e = np.ones(n_trace)
            # index, _ = np.where((e - m[i].detach().cpu().numpy()) == 0)
            d = int(n / 2)  # 缺失半径
            c = random.randint(d,n_trace-1-d)  # 缺失迹线中点
            index = list(set(range(0, n_trace)) - set(range(c - d, c + d)))
            M[0, index, index] = 1
        # return M

        return torch.tensor(M, dtype=torch.float, requires_grad=False)

def FKWeight(a, b, sc, st, fan_dip=1., filter_type='dip'):
    k1 = torch.linspace(-1., 1., sc)
    if filter_type == 'dip':
        weight = 0.5 * (1 + a + (a - 1) * torch.tanh(b * (k1)))
        weight = weight.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
    elif filter_type == 'fan_above' or filter_type == 'fan_under':
        weight = torch.zeros([sc, st // 2 + 1])
        for f in range(0,st // 2):
            d = (fan_dip * (f - 30)) / sc if f > 30 else 0
            w = 0.5 * (1 + a + (a - 1) * torch.tanh(b * (k1 + d)))
            weight[:, f] = w
        weight = weight + torch.flip(weight, dims=(0,))
        if filter_type == 'fan_under':
            weight = 1 - weight
        weight = weight.unsqueeze(0).unsqueeze(0)
    return weight

def FKfilter(input_tx, weight, compute_inverse=False):
    input_tx_t_AtTheEnd = input_tx.permute(0, 1, 3, 2)
    fk1 = torch.fft.fftshift(torch.fft.rfftn(input_tx_t_AtTheEnd), dim=(-2,))
    if compute_inverse:
        weight = 1.0 / weight
    weight_complex = torch.complex(weight, torch.zeros_like(weight))
    fk_filtered = torch.mul(fk1, weight_complex)
    output_tx_t_AtTheEnd = torch.fft.irfftn(torch.fft.ifftshift(fk_filtered, dim=(-2,)))
    return output_tx_t_AtTheEnd.permute(0, 1, 3, 2)


class CustomFKFilterLayer(nn.Module):
    def __init__(self, weight, compute_inverse=False):
        super(CustomFKFilterLayer, self).__init__()
        self.weight = weight
        self.compute_inverse = compute_inverse

    def forward(self, input_tx):
        input_tx_t_AtTheEnd = input_tx.permute(0, 1, 3, 2)
        fk1 = torch.fft.fftshift(torch.fft.rfftn(input_tx_t_AtTheEnd), dim=(-2,))

        if self.compute_inverse:
            self.weight = 1.0 / self.weight
        weight_complex = torch.complex(self.weight, torch.zeros_like(self.weight))
        fk_filtered = torch.mul(fk1, weight_complex)
        # a = fk_filtered.cpu().numpy()
        # a = np.abs(a)
        # # 显示图像
        # fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        # axs.imshow(a[0][0], cmap='jet')
        # axs.set_title('negative K suppressed')
        # axs.set_xlabel('Frequency')
        # axs.set_ylabel('Normalized wavenumber')
        # # plt.colorbar()
        # plt.show()
        output_tx_t_AtTheEnd = torch.fft.irfftn(torch.fft.ifftshift(fk_filtered, dim=(-2,)))
        output_tx = output_tx_t_AtTheEnd.permute(0, 1, 3, 2)
        return output_tx
#一维傅里叶变换
def AS(f):
    k1 = torch.linspace(-1., 1., 192)
    # fs = 10
    signal = torch.sin(2 * torch.pi * f * k1)
    x = torch.fft.fft(signal)
    m = torch.abs(x)
    return m
def fk1_weight(f, fp):
    s = (f - fp) / 4
    a = AS(f) / AS(fp)
    p = a + (1 - a * a) / a * torch.exp(-(1 - f) * (1 - f) / (2 * s * s))
    weight = p.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
    return weight


class FKfilterLayer1(nn.Module):
    def __init__(self, weight, compute_inverse=False, **kwargs):
        super(FKfilterLayer1, self).__init__(**kwargs)
        self.weight = weight
        self.compute_inverse = compute_inverse

    def forward(self, input_tx):
        input_tx = input_tx.permute(0, 1, 3, 2)
        fk1 = torch.fft.fftshift(torch.fft.fft(input_tx, dim=-2))
        if self.compute_inverse:
            self.weight = 1.0 / self.weight
        weight_complex = torch.complex(self.weight, torch.zeros_like(self.weight))
        fk_filtered = torch.mul(fk1, weight_complex)
        # a = fk_filtered.cpu().numpy()
        # a = np.abs(a)
        # # 显示图像
        # fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        # axs.imshow(a[0][0], cmap='jet')
        # axs.set_title('fk1')
        # axs.set_xlabel('Frequency')
        # axs.set_ylabel('Normalized wavenumber')
        # plt.show()
        output_tx = torch.fft.irfft(torch.fft.ifftshift(fk_filtered, dim=-2), fk1.shape[3])
        output_tx = output_tx.permute(0, 1, 3, 2)

        return output_tx

def gamma(r, x, compute_inverse=False):
    if compute_inverse:
        r = 1.0 / r
    g = torch.sign(x) * abs(x) ** r
    return g



if __name__ == '__main__':
    from math import ceil
    x = np.load('E:/muyuting/data/AVOclean.npy')[0]
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    mask1 = np.load('E:/muyuting/data/mask_AVO_36_2.npy')[0]
    mask1 =torch.from_numpy(mask1)
    mask1 = mask1.unsqueeze(0)


    # print(mask.shape)
    print(mask1.shape)
    # x = x.dot(mask)
    M = mask_sampling1(mask1, 0.3, '2')
    # np.save('E:/muyuting/data/mask_AVO_90_0.npy', y)
    y = x.matmul(M)
    plt.figure()
    plt.imshow(y[0], cmap=plt.cm.seismic, aspect=0.2, vmin=-1, vmax=1)
    # plt.imshow(y, cmap='jet', aspect=0.2, vmin=-1, vmax=1)
    plt.show()
    # x1 = np.load('E:/muyuting/data/AVOclean.npy')[0]
    # x1 = np.sign(x1) * np.log10(abs(x1*1) + 1)
    # x1 = torch.from_numpy(x1)
    # z = x1.matmul(y[0])


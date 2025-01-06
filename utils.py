import numpy as np
import random
import torch
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

def damage_operator(data, save_dir=None, rato=0.5, state=1):
    b,h,w = data.shape
    rato = int(w - w*rato)
    mask = np.zeros([b,w,w], dtype=np.float32)
    for i in range(b):
        mask[i,:,:] = mask_generator(w, rato, state)
    if save_dir is not None:
        np.save(save_dir+'_'+str(rato)+'_'+str(state), mask)
    return mask

def mask_generator(w, rato, state):
    # state: 0-'regularly' 1-'irregularly' 2-'continuously'
    mask = np.zeros([1,w,w], dtype=np.float32)
    if state == 0:
        index = list(range(0,w,int(w/rato))) #
    elif state == 1:
        index = np.int32(random.sample(range(0,w),rato))
    elif state == 2:
        d = int(rato/2)
        # print(d)# 缺失半径
        c = random.randint(d,w-1-d)
        # print(c)# 缺失迹线中点
        index = list(set(range(0,w))-set(range(c-d,c+d)))
    else:
        print('Don\'t have this state.')
    mask[0,index,index] = 1
    return mask

def mask_sampling(m, rato=0.05, missing_type='1'):
    # m: [B,H,W]
    # rato: non-missing rate
    n_trace = m.shape[-1]
    n = int(n_trace*rato)

    M = np.zeros(m.shape)
    if missing_type == '1':
        e = np.eye(n_trace)
        for i in range(len(m)):
            index,_ = np.where((e-m[i].detach().numpy())==1)
            # index, _ = np.where((e - m[i].numpy()) == 1)
            
            samples = random.sample(range(0, len(index)), n)
            indexR = index[samples]
            M[i,indexR,indexR] = 1
        
        return torch.tensor(M, dtype=torch.float, requires_grad=False)
    elif missing_type == '0':
        for i in range(len(m)):
            # index = random.sample(range(1, int(n_trace / n)), 1)[0]
            # samples = list(range(index, n_trace, int(n_trace / n)))
            #
            samples = random.sample(range(0, n_trace), n)
            M[i,samples,samples] = 1
        return torch.tensor(M, dtype=torch.float, requires_grad=False)
    else:
        pass

def Patchsubfunction(n, patch, H):
    # n: 划份份数；patch：切割大小；H：总长度
    if n*patch < H:
        print('The setting of n is wrong ')
    else:
        c = (n*patch-H)
        d = n-1
        a = int(c/d)
        x = (a+1)*d-c
        y = c-a*d
        i = 0
        out = [i]
        overlap = []
        for _ in range(x):
            i += patch-a
            out.append(i)
            overlap.append(a)
        for _ in range(y):
            i += patch-(a+1)
            out.append(i)
            overlap.append(a+1)
        return out,overlap
    
if __name__ == '__main__':
    # a = np.ones((1, 6, 6), dtype=np.float32)
    #
    # mask = mask_generator(w=6, rato=3, state=0)
    # print(mask)
    # data = torch.randn([1001, 1500, 120])
    # mask = damage_operator(data, save_dir='D:/data/mask_AVO', rato=0.7, state=2)
    # print(mask[:5])
    x = np.load('D:\data\syntheticdata\BPclean\clean1.npy')
    print(x.shape)
    # mask = np.load('D:/data/mask_AVO_36_2.npy')[0]
    # print(mask.shape)
    # x = x.dot(mask)
    # print(x.shape)
    x = np.sign(x) * np.log10(np.abs(x) + 1)
    plt.figure()
    plt.imshow(x, cmap=plt.cm.seismic, aspect=0.2)
    plt.show()
    # print(x.shape)













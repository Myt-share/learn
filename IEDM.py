import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
from module.down_unet import blindspotUNet as UNet
import matplotlib.pyplot as plt
from utils1 import *
import argparse


class Solver(object):
    def __init__(self, args):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda:0' if use_cuda else 'cpu'
        self.print_iter = args.print_iter
        self.max_iter = args.max_iter
        self.global_iter = 0
        self.dataset = args.dataset

        # Dataset
        self.data_loader = return_data(args)

        # Networks & Optimizers
        self.model = UNet(blindspot=True, method='Ours', dataset=args.dataset).to(self.device)
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.optim_model = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2))
        self.nets = [self.model]

        # Checkpoint
        self.ckpt_dir = args.ckpt_dir
        mkdirs(self.ckpt_dir)
        self.ckpt_save_iter = args.ckpt_save_iter
        self.save_image = args.save_image
        mkdirs(self.save_image)
        if args.ckpt_load:
            self.load_checkpoint(args.ckpt_load)


    def train(self):
        self.net_mode(train=True)

        out = False
        while not out:
            self.global_iter += 1
            _loss = 0

            for i, (x, m) in enumerate(self.data_loader):
                x = x.unsqueeze(1).float().to(self.device)
                m1 = m.unsqueeze(1).float().to(self.device)
                e = torch.eye(x.size(-1)).expand((x.size(0), 1, x.size(-1), x.size(-1))).to(self.device)
                for j in range(1):
                    M = mask_sampling(m, 0.2, '1').unsqueeze(1).float().to(self.device)
                    x1 = x.matmul(M)

                    x_recon = self.model(x1)
                    x_rafine = self.model(x_recon.matmul(e-m1) + x_recon.matmul(m1-m1.matmul(M)))

                    loss = (x_recon.matmul(m1-m1.matmul(M)) - x.matmul(m1-m1.matmul(M))).norm(1) / x.shape[0]
                    loss1 = (x_recon.matmul(m1) - x).norm(1) / x.shape[0]
                    loss2 = (x_rafine.matmul(m1.matmul(M)) - x1).norm(1) / x.shape[0]

                    Aloss = loss + loss1 + 0.15 * loss2

                    self.optim_model.zero_grad()
                    Aloss.backward()
                    self.optim_model.step()

                    _loss += Aloss.item()
            if self.global_iter % self.print_iter == 0:
                print('[{}] Aloss:{:.7f}'.format(self.global_iter, _loss / (i + 1)))
                self.save_images(x_recon, self.save_image + r'\recon1_epoch_%d.png' % self.global_iter)

            if self.global_iter % self.ckpt_save_iter == 0:
                self.save_checkpoint(self.global_iter)

            if self.global_iter >= self.max_iter:
                out = True


    def net_mode(self, train):
        # `Dropout`,`BatchNorm`,etc.
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def load_checkpoint(self, ckptname='last'):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                print("no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.model.load_state_dict(checkpoint['model_states']['model'])
            self.optim_model.load_state_dict(checkpoint['optim_states']['optim_model'])

    def save_checkpoint(self, ckptname='last'):
        model_states = {'model': self.model.state_dict()}
        optim_states = {'optim_model': self.optim_model.state_dict()}
        states = {'iter': self.global_iter,
                  'model_states': model_states,
                  'optim_states': optim_states}
        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def save_images(self, imgs, root):
        imgs = imgs.squeeze(1)
        im = imgs[0]
        im = im.cpu().detach().numpy()
        plt.imshow(im, cmap='gray', interpolation='nearest', aspect=0.8, vmin=-0.5, vmax=0.5)
        plt.savefig(root)



def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='work')
    parser.add_argument('--cuda', default=True, type=bool, help='enable cuda')
    # parser.add_argument('--dataset', default='MAVGL12', type=str, help='dataset name')  # MAVGL12
    # parser.add_argument('--data_dir', default=r'E:/muyuting/data/AVOclean.npy', type=str, help='dataset directory')#M94_clean,M94_mask_1_75
    # parser.add_argument('--mask_dir', default=r'E:/muyuting/data/mask_AVO_30_1.npy', type=str, help='dataset directory')

    parser.add_argument('--dataset', default='M94', type=str, help='dataset name')
    parser.add_argument('--data_dir', default=r'E:/muyuting/data/M94_clean.npy', type=str,
                        help='dataset directory')  # M94_clean,M94_mask_1_75
    parser.add_argument('--mask_dir', default=r'E:/muyuting/data/M94_mask_1_75.npy', type=str, help='dataset directory')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    # parser.add_argument('--obratio', default=0.5, type=float, help='-')
    parser.add_argument('--method', default='our', type=str)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate of the model')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the model')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='beta2 parameter of the Adam optimizer for the model')
    parser.add_argument('--max_iter', default=100, type=float, help='maximum training iteration')
    parser.add_argument('--print_iter', default=1, type=int, help='print losses iter')
    parser.add_argument('--ckpt_dir', default=r'E:/muyuting/EDMLearing//checkpoint', type=str,
                        help='checkpoint directory')
    parser.add_argument('--ckpt_save_iter', default=1, type=float, help='checkpoint save iter')
    parser.add_argument('--save_image', default=r'E:/muyuting/EDMLearing/reconstruct', type=str,
                        help='reconstruct data')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    args = parser.parse_args()
    dataset = SeismicData(args.data_dir, args.mask_dir)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True)
    solver = Solver(args)
    solver.load_checkpoint()
    solver.train()

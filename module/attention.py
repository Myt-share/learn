import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # self.channels = channel
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        out1 = self.pool1(x)
        out2 = self.pool2(x)
        out1 = self.conv2(out1)
        out2 = self.conv2(out2)
        out = self.sig(out1 + out2) * x
        # out = self.relu(out)
        return out
'''

class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()
        self.channels = channel
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.conv2 = nn.Conv2d(self.channels * 4, 1, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(self.channels * 4, self.channels*2, kernel_size=1, stride=1)



    def forward(self, x):
        out1 = self.pool1(x)
        out2 = self.pool2(x)
        out1 = self.conv2(out1)
        out2 = self.conv2(out2)
        out = self.sig(out1 + out2) * x
        out = self.conv(out)

        return out'''

class attention1(nn.Module):
    def __init__(self):
        super(attention1, self).__init__()
        self.conv1 = nn.Conv2d(512, 64, kernel_size=3, stride=1)
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool3 = nn.AvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()
        self.lin1 = nn.Linear(95, 95)
        self.relu = nn.ReLU()


    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.pool1(out1)
        out3 = self.pool2(out1)
        out = torch.cat([out2, out3], dim=1)
        out4 = self.conv2(out)
        out5 = self.sig(out4) * x
        sout = self.pool3(x)
        sout1 = self.lin1(sout)
        sout1 = self.relu(sout1)
        sout1 = self.lin1(sout1)
        sout1 = self.sig(sout1) * x
        output = sout1 + out5
        return output

class attention2(nn.Module):
    def __init__(self):
        super(attention2, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1)
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.pool3 = nn.AvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.sig = nn.Sigmoid()
        self.lin1 = nn.Linear(25, 25)
        # self.lin1 = nn.Linear(47, 47)
        self.relu = nn.ReLU()


    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.pool1(out1)
        out3 = self.pool2(out1)
        out4 = self.conv2(out2)
        out5 = self.conv2(out3)
        out6 = self.sig(out4+out5) * x
        sout = self.pool3(x)
        sout1 = self.lin1(sout)
        sout1 = self.relu(sout1)
        sout1 = self.lin1(sout1)
        sout1 = self.sig(sout1) * x
        output = sout1 + out6
        return output



#下采样
class ConvTransBlock(nn.Module):
    def __init__(self, channel, conv_dim, trans_dim):
        """ Swin-Transformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.channel = channel
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.conv1 = nn.Conv2d(self.channel, self.conv_dim + self.trans_dim, 1, 2,  bias=True)
        self.sig = nn.Sigmoid()
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 1, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, 2*self.conv_dim, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(2*self.conv_dim, self.conv_dim, 1, 1, bias=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x1 = self.pool1(conv_x)
        conv_x1 = self.conv_block(conv_x1)
        conv_x1 = self.sig(conv_x1) * conv_x
        # res = self.conv1_2(torch.cat((conv_x1, trans_x), dim=1))
        res = torch.cat((conv_x1, trans_x), dim=1)
        return res
#通道融合
class LayerNorm(nn.Module):
    def __init__(self,  eps=1e-6):
        super().__init__()
        self.normalized_shape = [9, 95]
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps = eps


    def forward(self, x):

            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class CCM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.ln = LayerNorm()
        self.ccm = nn.Sequential(
            nn.Conv2d(self.channel, self.channel * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(self.channel * 2, self.channel, 1, 1, 0)
        )

    def forward(self, x):
        x = self.ln(x)
        out = self.ccm(x)
        return out

if __name__ == '__main__':
    x = torch.rand(2, 1024, 9, 95)
    # print(x[0].shape)
    # model = attention2()
    # model = Attention(256)
    # out = model(x)
    # print('out', out.shape)


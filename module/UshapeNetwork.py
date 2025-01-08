import torch
import torch.nn as nn
from torch import Tensor
from module.component import ShiftConv2d, Shift2d, Crop2d, rotate
from utils1 import *
from module.attention import CCM
'''
def _max_pool_block(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat([x_LL, x_HL, x_LH, x_HH], 1)
def _up_sampling(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return _max_pool_block(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return _up_sampling(x)







class blindspotUNet(nn.Module):

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            blindspot: bool = False,
            zero_output_weights: bool = False,
            method=None,
            dataset=None
    ):
        super(blindspotUNet, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights
        self.Conv2d = ShiftConv2d if self.blindspot else nn.Conv2d
        filters = [32, 64, 128, 256]
        self.dwt = DWT()
        self.iwt = IWT()
        self.td = CCM(filters[3])

        ####################################
        # Encode Blocks
        ####################################

        self.encode_block_1 = nn.Sequential(
            self.Conv2d(in_channels, filters[0], 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0], affine=True),
            nn.ReLU(inplace=True),
            self.Conv2d(filters[0], filters[0], 3, padding=1, bias=True),
            nn.BatchNorm2d(filters[0], affine=True),
            nn.ReLU(inplace=True),
        )



        def _encode_block_2_3_4(channels) -> nn.Module:

            return nn.Sequential(
                self.Conv2d(channels * 2, channels, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(channels, affine=True),
                nn.ReLU(inplace=True),
                self.Conv2d(channels, channels, 3, padding=1, bias=True),
                nn.BatchNorm2d(channels, affine=True),
                nn.ReLU(inplace=True),
            )

        self.encode_block_2 = _encode_block_2_3_4(filters[1])
        self.encode_block_3 = _encode_block_2_3_4(filters[2])
        self.encode_block_4 = _encode_block_2_3_4(filters[3])

        ####################################
        # Decode Blocks
        ####################################


        self.bottleneck = nn.Sequential(

            self.Conv2d(filters[3]*4, filters[3] * 4, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[3] * 4, affine=True),
            nn.ReLU(inplace=True),
            self.Conv2d(filters[3] *4 , filters[3] * 4, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters[3] * 4, affine=True),
            nn.ReLU(inplace=True),
            # _up_sampling(filters[3] * 2),
        )

        def _decode_block4_3_2(channels) -> nn.Module:
            return nn.Sequential(
                self.Conv2d(channels, channels * 2, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(channels * 2, affine=True),
                nn.ReLU(inplace=True),
                self.Conv2d(channels * 2, channels * 2, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(channels * 2, affine=True),
                nn.ReLU(inplace=True),
                # _up_sampling(channels),
            )

        self.decode_block_4 = _decode_block4_3_2(filters[3])
        self.decode_block_3 = _decode_block4_3_2(filters[2])
        self.decode_block_2 = _decode_block4_3_2(filters[1])

        self.decode_block_1 = nn.Sequential(
            self.Conv2d(filters[0], filters[0], 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0], affine=True),
            nn.ReLU(inplace=True),
            self.Conv2d(filters[0], filters[0], 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0], affine=True),
            nn.ReLU(inplace=True),
        )

        ####################################
        # Output Block
        ####################################

        if self.blindspot:
            # Shift 1 pixel down
            self.shift = Shift2d((1, 0))
            # 2 x Channels due to batch rotations
            nin_a_io = 64
        else:
            nin_a_io = 32

        self.output_conv = self.Conv2d(32, out_channels, 1)
        self.output_block = nn.Sequential(
            self.Conv2d(nin_a_io, 32, 1),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            self.output_conv,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        if dataset == 'MAVGL12':
            if method == 'DSP':
                self.crop5 = Crop2d((1, 0, 1, 0))
                self.crop4 = Crop2d((1, 1, 1, 0))
                self.crop3 = Crop2d((1, 0, 1, 1))
                self.crop2 = Crop2d((1, 0, 1, 0))
                self.crop1 = Crop2d((1, 1, 1, 1))
            else:
                self.crop5 = Crop2d((1, 0, 1, 0))
                self.crop4 = Crop2d((1, 0, 1, 1))
                self.crop3 = Crop2d((1, 1, 1, 0))
                self.crop2 = Crop2d((1, 0, 1, 0))
                self.crop1 = Crop2d((1, 1, 1, 1))
        else:
            if method == 'DSP':
                self.crop5 = Crop2d((1, 0, 1, 0))
                self.crop4 = Crop2d((1, 0, 1, 0))
                self.crop3 = Crop2d((1, 0, 1, 0))
                self.crop2 = Crop2d((1, 0, 1, 0))
                self.crop1 = Crop2d((1, 1, 1, 1))
            else:

                self.crop4 = Crop2d((1, 0, 1, 0))
                self.crop3 = Crop2d((1, 0, 1, 0))
                self.crop2 = Crop2d((1, 0, 1, 0))
                self.crop1 = Crop2d((1, 1, 1, 1))
        # Initialize weights
        self.init_weights()
        self.c1 = Crop2d((0, 1, 0, 0))
        self.c2 = Crop2d((0, 1, 0, 1))

    @property
    def blindspot(self) -> bool:
        return self._blindspot

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        # Initialise last output layer
        if self._zero_output_weights:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x, mask=None):
        if self.blindspot:
            rotated = [rotate(x, rot) for rot in (90, 270)]
            x = torch.cat((rotated), dim=0)

        # Encoder
        enc1 = self.encode_block_1(x)

        enc1 = self.dwt(enc1)
        enc2 = self.encode_block_2(enc1)
        enc2 = self.dwt(enc2)
        enc3 = self.encode_block_3(enc2)
        enc3 = self.dwt(enc3)
        enc4 = self.encode_block_4(enc3)
        enc4 = self.dwt(enc4)
        print(enc4.shape)
        enc4 = self.td(enc4)
        # Decoder
        upsample4 = self.bottleneck(enc4)
        upsample3 = self.decode_block_4(self.iwt(upsample4)) + enc3
        upsample2 = self.decode_block_3(self.iwt(upsample3)) + enc2
        upsample1 = self.decode_block_2(self.iwt(upsample2)) + enc1
        x = self.decode_block_1(self.iwt(upsample1))
        # print(x.shape)

        # Output
        if self.blindspot:
            # Apply shift
            shifted = self.shift(x)
            # Unstack, rotate and combine
            rotated_batch = torch.chunk(shifted, 2, dim=0)
            aligned = [
                rotate(rotated, rot)
                for rotated, rot in zip(rotated_batch, (270, 90))
            ]
            x = torch.cat(aligned, dim=1)

        x = self.output_block(x)
        return x

    @staticmethod
    def input_wh_mul() -> int:

        max_pool_layers = 5
        return 2 ** max_pool_layers

'''


class blindspotUNet(nn.Module):


    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        blindspot: bool = False,
        zero_output_weights: bool = False,
        method=None,
        dataset=None
        ):
        super(blindspotUNet, self).__init__()
        self._blindspot = blindspot
        self._zero_output_weights = zero_output_weights
        self.Conv2d = ShiftConv2d if self.blindspot else nn.Conv2d
        filters = [32, 64, 128, 256, 512]
        self.ccm1 = CCM(filters[0])
        self.ccm2 = CCM(filters[1])
        self.ccm3 = CCM(filters[2])
        self.ccm4 = CCM(filters[3])
        self.ccm5 = CCM(filters[4])

        ####################################
        # Encode Blocks
        ####################################

        def _max_pool_block(max_pool: nn.Module) -> nn.Module:
            if blindspot:
                return nn.Sequential(Shift2d((1, 0)), max_pool)
            return max_pool

        self.encode_block_1 = nn.Sequential(
            self.Conv2d(in_channels, filters[0], 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0],affine=True),
            nn.ReLU(inplace=True),
            self.Conv2d(filters[0], filters[0], 3, padding=1, bias=True),
            nn.BatchNorm2d(filters[0],affine=True),
            nn.ReLU(inplace=True),
            )

        def _encode_block_2_3_4_5(channels) -> nn.Module:
            return nn.Sequential(
                _max_pool_block(nn.MaxPool2d(2, padding=1)),
                self.Conv2d(channels//2, channels, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(channels,affine=True),
                nn.ReLU(inplace=True),
                self.Conv2d(channels, channels, 3, padding=1, bias=True),
                nn.BatchNorm2d(channels,affine=True),
                nn.ReLU(inplace=True),
                )

        self.encode_block_2 = _encode_block_2_3_4_5(filters[1])
        self.encode_block_3 = _encode_block_2_3_4_5(filters[2])
        self.encode_block_4 = _encode_block_2_3_4_5(filters[3])
        self.encode_block_5 = _encode_block_2_3_4_5(filters[4])

        ####################################
        # Decode Blocks
        ####################################

        def _up_sampling(channels):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 self.Conv2d(channels, channels//2, kernel_size=3, stride=1, padding=1, bias=True),
                                 nn.BatchNorm2d(channels//2,affine=True),
                                 nn.ReLU(inplace=True))

        self.bottleneck = nn.Sequential(
            _max_pool_block(nn.MaxPool2d(2, padding=1)),
            self.Conv2d(filters[4], filters[4]*2, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[4]*2,affine=True),
            nn.ReLU(inplace=True),
            self.Conv2d(filters[4]*2, filters[4]*2, 3, padding=1, bias=True),
            nn.BatchNorm2d(filters[4]*2,affine=True),
            nn.ReLU(inplace=True),
            _up_sampling(filters[4]*2),
            )

        def _decode_block_5_4_3_2(channels) -> nn.Module:
            return nn.Sequential(
                self.Conv2d(channels*2, channels, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(channels,affine=True),
                nn.ReLU(inplace=True),
                self.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(channels,affine=True),
                nn.ReLU(inplace=True),
                _up_sampling(channels),
                )

        self.decode_block_5 = _decode_block_5_4_3_2(filters[4])
        self.decode_block_4 = _decode_block_5_4_3_2(filters[3])
        self.decode_block_3 = _decode_block_5_4_3_2(filters[2])
        self.decode_block_2 = _decode_block_5_4_3_2(filters[1])

        self.decode_block_1 = nn.Sequential(
            self.Conv2d(filters[1], filters[0], 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0],affine=True),
            nn.ReLU(inplace=True),
            self.Conv2d(filters[0], filters[0], 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filters[0],affine=True),
            nn.ReLU(inplace=True),
            )

        ####################################
        # Output Block
        ####################################

        if self.blindspot:
            # Shift 1 pixel down
            self.shift = Shift2d((1, 0))
            # 2 x Channels due to batch rotations
            nin_a_io = 64
        else:
            nin_a_io = 32

        self.output_conv = self.Conv2d(32, out_channels, 1)
        self.output_block = nn.Sequential(
            self.Conv2d(nin_a_io, 32, 1),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            self.output_conv,
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        if dataset == 'MAVGL12':
            if method == 'DSP':
                self.crop5 = Crop2d((1,0,1,0))
                self.crop4 = Crop2d((1,1,1,0))
                self.crop3 = Crop2d((1,0,1,1))
                self.crop2 = Crop2d((1,0,1,0))
                self.crop1 = Crop2d((1,1,1,1))
            else:
                self.crop5 = Crop2d((1,0,1,0))
                self.crop4 = Crop2d((1,0,1,1))
                self.crop3 = Crop2d((1,1,1,0))
                self.crop2 = Crop2d((1,0,1,0))
                self.crop1 = Crop2d((1,1,1,1))
        else:
            if method == 'DSP':
                self.crop5 = Crop2d((1,0,1,0))
                self.crop4 = Crop2d((1,0,1,0))
                self.crop3 = Crop2d((1,0,1,0))
                self.crop2 = Crop2d((1,0,1,0))
                self.crop1 = Crop2d((1,1,1,1))
            else:
                self.crop5 = Crop2d((1,0,1,0))
                self.crop4 = Crop2d((1,0,1,0))
                self.crop3 = Crop2d((1,0,1,0))
                self.crop2 = Crop2d((1,0,1,0))
                self.crop1 = Crop2d((1,1,1,1))
        # Initialize weights
        self.init_weights()

    @property
    def blindspot(self) -> bool:
        return self._blindspot

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).
        """
        with torch.no_grad():
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.1)
                m.bias.data.zero_()
        # Initialise last output layer
        if self._zero_output_weights:
            self.output_conv.weight.zero_()
        else:
            nn.init.kaiming_normal_(self.output_conv.weight.data, nonlinearity="linear")

    def forward(self, x, mask=None):
        if self.blindspot:
            rotated = [rotate(x, rot) for rot in (90, 270)]
            x = torch.cat((rotated), dim=0)

        # Encoder
        enc1 = self.encode_block_1(x)
        # enc1 = self.ccm1(enc1)
        enc2 = self.encode_block_2(enc1)
        # enc2 = self.ccm2(enc2)
        enc3 = self.encode_block_3(enc2)
        # enc3 = self.ccm3(enc3)
        enc4 = self.encode_block_4(enc3)
        # enc4 = self.ccm4(enc4)
        enc5 = self.encode_block_5(enc4)
        enc5 = self.ccm5(enc5)

        # Decoder
        UP = self.bottleneck(enc5)
        upsample5 = self.crop5(UP)
        concat5 = torch.cat((upsample5, enc5), dim=1)
        upsample4 = self.crop4(self.decode_block_5(concat5))
        concat4 = torch.cat((upsample4, enc4), dim=1)
        upsample3 = self.crop3(self.decode_block_4(concat4))
        concat3 = torch.cat((upsample3, enc3), dim=1)
        upsample2 = self.crop2(self.decode_block_3(concat3))
        concat2 = torch.cat((upsample2, enc2), dim=1)
        upsample1 = self.crop1(self.decode_block_2(concat2))
        concat1 = torch.cat((upsample1, enc1), dim=1)
        x = self.decode_block_1(concat1)

        # Output
        if self.blindspot:
            # Apply shift
            shifted = self.shift(x)
            # Unstack, rotate and combine
            rotated_batch = torch.chunk(shifted, 2, dim=0)
            aligned = [
                rotate(rotated, rot)
                for rotated, rot in zip(rotated_batch, (270, 90))
            ]
            x = torch.cat(aligned, dim=1)

        x = self.output_block(x)
        return x

    @staticmethod
    def input_wh_mul() -> int:
        """Multiple that both the width and height dimensions of an input must be to be
        processed by the network. This is devised from the number of pooling layers that
        reduce the input size.
        Returns:
            int: Dimension multiplier
        """
        max_pool_layers = 5
        return 2 ** max_pool_layers


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    a = torch.randn(1, 1, 1500, 120).to(device)
    model = blindspotUNet(blindspot=True, method='our', dataset='MAVGL12').to(device)
    y = model(a)
    print(y.shape)
    # x = torch.randn(1024, 1024, 8, 194)
    # y = _up_sampling(x)
    # print(y.shape)

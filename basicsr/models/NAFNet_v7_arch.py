# ------------------------------------------------------------------------
# Copyright (c) 2022 Murufeng. All Rights Reserved.
# ------------------------------------------------------------------------
'''
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
        n_div: int = 4,
):

    return nn.Sequential(Partial_conv3(in_channels, n_div = n_div, kernel_size = kernel_size, padding = padding),
                        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))
    

class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, kernel_size= 3, padding = 1, forward='slicing'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, padding, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., n_div = 4):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,bias=True)
        #print("dw : ", dw_channel)
        self.conv2 = Partial_conv3(dw_channel, n_div = n_div, kernel_size = 3, padding = 1)
#        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv3 = nn.Sequential(Partial_conv3(dw_channel //2, n_div = n_div, kernel_size = 1, padding = 0),
                        nn.Conv2d(dw_channel//2, c, 1, 1, 0, bias=True)
                    )
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Sequential(Partial_conv3(ffn_channel //2, n_div = n_div, kernel_size = 1, padding = 0),
                        nn.Conv2d(ffn_channel//2, c, 1, 1, 0, bias=True)
                    )

#        self.norm1 = LayerNorm2d(c)
#        self.norm2 = LayerNorm2d(c)
        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)

        if drop_out_rate>0.:
            self.flag = True
        else:
            self.flag = False
        if self.flag:
            self.dropout1 = nn.Dropout(drop_out_rate)
            self.dropout2 = nn.Dropout(drop_out_rate) 

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        if self.flag:
            x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        if self.flag:
            x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet_v7(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for idx, num in enumerate(enc_blk_nums):
            n_div = 2 if idx==2 else 4
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, n_div = n_div) for _ in range(num)]
                )
            )
            '''
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2),
            )
            '''
            self.downs.append(
                nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                Partial_conv3(chan, n_div = 4, kernel_size = 3, padding = 1),
#                nn.ReLU(inplace=True),
                nn.Conv2d(chan, 2*chan, 1, 1, 0, bias=False)
                )
            ) 
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for idx, num in enumerate(dec_blk_nums):
            n_div = 2 if idx ==0 else 4
            '''
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            '''
            self.ups.append(
                nn.Sequential(
                    Conv2D(chan, chan//2, kernel_size = 3, stride=1, padding=1),
                    nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = False)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, n_div = n_div) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    img_channel = 3
    '''
    width = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    '''
    width = 16
    enc_blks = [1, 1, 1]
    middle_blk_num = 1
    dec_blks = [1, 1, 1]
    
    net = NAFNet_v7(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info
    from thop import profile
    data = torch.randn(1,3,256,256)
    flops, params = profile(net, inputs = (data,))
    print("FLOPs=",flops/(10**9),"G,","Parameters=",params/(10**3),"K")

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)


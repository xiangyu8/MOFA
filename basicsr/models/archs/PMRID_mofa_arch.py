# from https://github.com/bigeagle/PMRID/blob/main/models/net_torch.py
#!/usr/bin/env python3
import torch
import torch.nn as nn
from collections import OrderedDict

import numpy as np
from torch import Tensor
from  basicsr.models.archs.local_arch import Local_Base

def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
        n_div: int = 4,
        skip: bool = False
):
    modules = OrderedDict()

    if is_seperable and stride==1:
#        if stride==2:
#            modules['down'] = nn.AvgPool2d(kernel_size = 2, stride=2)
        dw_status = True if in_channels//n_div>=32  else False
        modules['pconvdw'] = pconv(in_channels = in_channels,out_channels = out_channels, kernel_size = kernel_size, padding = padding, n_div = n_div, skip = skip, dw_status = dw_status)
    elif is_seperable:
        modules['sep'] = sep(in_channels, out_channels, kernel_size, stride, padding)
    else:
        modules['conv'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_relu:
        modules['relu'] = nn.ReLU()

    return nn.Sequential(modules)

def pconv(in_channels, out_channels, kernel_size, padding, n_div = 4, skip = False, dw_status = False):
    modules = OrderedDict()
    modules['depthwise'] = Partial_conv3(in_channels, n_div = n_div, kernel_size = kernel_size, padding = padding,skip = skip, dw_status = dw_status)
    modules['pointwise'] = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
    return nn.Sequential(modules)

def sep(in_channels, out_channels, kernel_size, stride, padding):
    modules = OrderedDict()
    modules['depthwise'] = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups = in_channels, bias=False)
    modules['pointwise'] = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
    return nn.Sequential(modules)


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, kernel_size= 3, padding = 1, forward='split_cat', skip = False, dw_status = False):
        super().__init__()
        self.dim_conv3 = dim // n_div
        #print(self.dim_conv3)
        self.dim_untouched = dim - self.dim_conv3
        if dw_status:
            self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, padding, groups=self.dim_conv3, bias=False)
        else:
            self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, padding, 1, bias=False)
        self.skip = skip

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        if self.skip:
            x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

class EncoderBlock(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1, n_div: int=4):
        super().__init__()
        
        skip = True if stride==1 and in_channels==out_channels else False
        self.conv1 = Conv2D(in_channels, mid_channels, kernel_size=5, stride=stride, padding=2, is_seperable=True, has_relu=True, n_div = n_div, skip = skip)
        self.conv2 = Conv2D(mid_channels, out_channels, kernel_size=5, stride=1, padding=2, is_seperable=True, has_relu=False, n_div = n_div)
        self.stride = stride
        self.proj = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels else
            Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, is_seperable=True, has_relu=False)
        )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        proj = self.proj(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + proj
        return self.relu(x)


def EncoderStage(in_channels: int, out_channels: int, num_blocks: int, n_div: int = 4):

    blocks = [
        EncoderBlock(
            in_channels=in_channels,
            mid_channels=out_channels//4,
            out_channels=out_channels,
            stride=2,
        )
    ]
    for _ in range(num_blocks-1):
        blocks.append(
            EncoderBlock(
                in_channels=out_channels,
                mid_channels=out_channels//4,
                out_channels=out_channels,
                stride=1,
                n_div = n_div,
            )
        )

    return nn.Sequential(*blocks)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, n_div: int = 4):
        super().__init__()

        padding = kernel_size // 2
        self.conv0 = Conv2D(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=True, n_div = n_div, skip = True
        )
        self.conv1 = Conv2D(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=False, n_div = n_div
        )

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = x + inp
        return x


class DecoderStage(nn.Module):

    def __init__(self, in_channels: int, skip_in_channels: int, out_channels: int, n_div: int = 4):
        super().__init__()

        self.decode_conv = DecoderBlock(in_channels, in_channels, kernel_size=3, n_div = n_div)
#        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        
        self.upsample = nn.Sequential(
            Conv2D(in_channels, out_channels, kernel_size = 3, stride=1, padding=1, is_seperable = True),
            nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = False)
        )
        
        self.proj_conv = Conv2D(skip_in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=True, has_relu=True, n_div = n_div)
        # M.init.msra_normal_(self.upsample.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, inp, skip):
#        inp, skip = inputs

        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)
        return x + y


class PMRID_mofa(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        self.conv0 = Conv2D(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=16, out_channels=64, num_blocks=2)
        self.enc2 = EncoderStage(in_channels=64, out_channels=128, num_blocks=2)
        self.enc3 = EncoderStage(in_channels=128, out_channels=256, num_blocks=4, n_div = 2)
        self.enc4 = EncoderStage(in_channels=256, out_channels=512, num_blocks=4)

        self.encdec = Conv2D(in_channels=512, out_channels=64, kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True,n_div=2)
        self.dec1 = DecoderStage(in_channels=64, skip_in_channels=256, out_channels=64, n_div = 2)
        self.dec2 = DecoderStage(in_channels=64, skip_in_channels=128, out_channels=32, n_div = 2)
        self.dec3 = DecoderStage(in_channels=32, skip_in_channels=64, out_channels=32, n_div = 2)
        self.dec4 = DecoderStage(in_channels=32, skip_in_channels=16, out_channels=16, n_div = 2)

        self.out0 = DecoderBlock(in_channels=16, out_channels=16, kernel_size=3)
        self.out1 = Conv2D(in_channels=16, out_channels=in_channels, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)

    def forward(self, inp):

        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)
        
#        print(conv5.shape, conv3.shape)
        up3 = self.dec1(conv5, conv3)
        up2 = self.dec2(up3, conv2)
        up1 = self.dec3(up2, conv1)
        x = self.dec4(up1, conv0)

        x = self.out0(x)
        x = self.out1(x)

        pred = inp + x
        return pred

class PMRIDMofaLocal(Local_Base, PMRID_mofa):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        PMRID_mofa.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == "__main__":

    model = PMRID_mofa(in_channels=3)
    model.eval()
    #print(model)
    input = torch.randn(1, 3, 256, 256)
    # input = torch.randn(1, 3, 32, 32)
    y = model(input)
    print(y.size())

    from thop import profile

    flops, params = profile(model=model, inputs=(input,))
    print('Model:{:.2f} GFLOPs and {:.2f}M parameters'.format(flops / 10**9, params / 1e6))

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)



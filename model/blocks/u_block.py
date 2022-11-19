import torch.nn as nn
import torch
from model.blocks.normalization import SPADE_Sc
from .upsample import PixelShufflePack
from model.generator.pnet import SResBlock

class Udown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, norm_fun=nn.BatchNorm2d):
        super(Udown, self).__init__()
        layers = [nn.LeakyReLU(0.2, True)]
        layers.append(nn.Conv2d(in_size, out_size, 4, 2, 1))
        if normalize:
            layers.append(norm_fun(out_size, track_running_stats=False))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SUp(nn.Module):
    def __init__(self, opt, in_size, out_size, dropout=0.0, first=False):
        super(SUp, self).__init__()
        parsing_nc = opt.parsing_nc

        spade_config_str = opt.norm_G.replace('spectral', '')

        layers = [nn.Tanh(),
                  make_layer(ResidualBlockNoBN, num_blocks=2, mid_channels=in_size),
                  nn.Conv2d(in_size, in_size,1,1,0),
                  PixelShufflePack(in_size, out_size, scale_factor=2, upsample_kernel=1),
                  ]


        if not first:
            self.norm = SPADE_Sc(spade_config_str, out_size, parsing_nc, opt.spade_mode, opt.use_en_feature)
        else:
            self.norm = SPADE_Sc(spade_config_str, out_size, parsing_nc, opt.spade_mode)
        self.en_conv = nn.ConvTranspose2d(in_size // 2, parsing_nc, 4, 2, 1)
        self.dp = None
        if dropout:
            self.dp = nn.Dropout(dropout)

        self.model = nn.Sequential(*layers)
        self.opt = opt

    def forward(self, de_in, parsing, en_in=None, gamma_mode='none'):
        x = de_in
        en_affine = None
        if en_in is not None:
            x = torch.cat([de_in, en_in], dim=1)
            if self.opt.use_en_feature:
                en_affine = self.en_conv(en_in)
        x = self.model(x)
        if gamma_mode != 'none':
            x, gamma_beta = self.norm(x, parsing, en_affine, gamma_mode=gamma_mode)
        else:
            x = self.norm(x, parsing, en_affine, gamma_mode=gamma_mode)
        if self.dp is not None:
            x = self.dp(x)
        if gamma_mode != 'none':
            return x, gamma_beta
        else:
            return x


from model.blocks.u_block import Udown, SUp
from .upsample import PixelShufflePack
import torch.nn as nn
from .sr_backbone import (ResidualBlockNoBN, default_init_weights, make_layer)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels=64, res_scale=1.0):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style
        modules. For modules with residual paths, using smaller std is
        better for stability and performance. We empirically use 0.1.
        See more details in "ESRGAN: Enhanced Super-Resolution Generative
        Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


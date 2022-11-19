

import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.spectral_norm as spectral_norm
# Returns a function that creates a normalization function
# that does not condition on semantic map
from model.blocks.sync_batchnorm import SynchronizedBatchNorm2d


def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer


        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer



class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        self.ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False, track_running_stats=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        self.nhidden = 128

        self.pw = self.ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, self.nhidden, kernel_size=self.ks, padding=self.pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(self.nhidden, norm_nc, kernel_size=self.ks, padding=self.pw)
        self.mlp_beta = nn.Conv2d(self.nhidden, norm_nc, kernel_size=self.ks, padding=self.pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        gamma, beta = self.get_spade_gamma_beta(normalized, segmap)
        out = normalized * (1 + gamma) + beta

        return out

    def get_spade_gamma_beta(self, normed, segmap):
        segmap = F.interpolate(segmap, size=normed.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return gamma, beta


class SPADE_Sc(SPADE):
    def __init__(self, config_text, norm_nc, label_nc, spade_mode="concat", use_en_feature=False, dil=False):
        super(SPADE_Sc, self).__init__(config_text, norm_nc, label_nc)
        self.spade_mode = spade_mode
        self.dil = dil
        self.label_nc = label_nc
        if spade_mode == 'concat':
            # concat + conv
            self.con_conv = nn.Conv2d(norm_nc * 2, norm_nc, kernel_size=self.ks, padding=self.pw)
        if use_en_feature:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(label_nc * 2, self.nhidden, kernel_size=self.ks, padding=self.pw),
                nn.ReLU()
            )

    def forward(self, x, segmap, en_feature=None, gamma_mode='none'):
        # spade org
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if en_feature != None:
            segmap = torch.cat([segmap, en_feature], dim=1)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        if self.spade_mode == 'concat':
            concating = torch.cat([normalized, out], dim=1)
            out = self.con_conv(concating)
        elif self.spade_mode == 'res':
            out = out + normalized
        elif self.spade_mode == 'res2':
            out = out + x

        if gamma_mode == 'final':
            gamma_beta = torch.cat([gamma, beta], dim=1)  # use to calc l1
            return out, gamma_beta
        elif gamma_mode == 'feature':
            norm_gamma = normalized.running_mean
            norm_beta = normalized.running_var
            gamma_beta = torch.cat([norm_gamma, norm_beta], dim=1)  # use to calc l1
            return out, gamma_beta
        else:
            return out

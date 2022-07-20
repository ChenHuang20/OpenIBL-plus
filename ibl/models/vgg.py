from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from ..utils.serialization import load_checkpoint, copy_state_dict

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(in_planes=channel)
        self.sa=SpatialAttention()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual

__all__ = ['VGG', 'vgg16']


class VGG(nn.Module):
    __factory = {
        16: torchvision.models.vgg16,
    }

    __fix_layers = { # vgg16
        'conv5':24,
        'conv4':17,
        'conv3':10,
        'conv2':5,
        'full':0
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                    train_layers='conv5', matconvnet=None):
        super(VGG, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.train_layers = train_layers
        self.feature_dim = 512
        self.matconvnet = matconvnet
        # Construct base (pretrained) resnet
        if depth not in VGG.__factory:
            raise KeyError("Unsupported depth:", depth)
        vgg = VGG.__factory[depth](pretrained=pretrained)
        layers = list(vgg.features.children())[:-2]
        self.base = nn.Sequential(*layers) # capture only feature part and remove last relu and maxpool
        # 增加两个注意力通道
        self.ca=ChannelAttention(in_planes=self.feature_dim)
        self.sa=SpatialAttention()
        self.gap = nn.AdaptiveMaxPool2d(1)

        self._init_params()

        if not pretrained:
            self.reset_params()
        else:
            layers = list(self.base.children())
            for l in layers[:VGG.__fix_layers[train_layers]]:
                for p in l.parameters():
                    p.requires_grad = False

    def _init_params(self):
        # optional load pretrained weights from matconvnet
        if (self.matconvnet is not None):
            self.base.load_state_dict(torch.load(self.matconvnet))
            self.pretrained = True

    def forward(self, x):
        x = self.base(x)

        # 注意力前向传播
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        x = out + residual

        if self.cut_at_pooling:
            return x

        pool_x = self.gap(x)
        pool_x = pool_x.view(pool_x.size(0), -1)

        return pool_x, x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

def vgg16(**kwargs):
    return VGG(16, **kwargs)
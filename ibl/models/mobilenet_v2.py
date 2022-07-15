import torch.nn as nn
import math
import torchvision
import torch


__all__ = ['mobilenetv2']
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

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
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)
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

    def __init__(self, channel=320,reduction=16,kernel_size=49):
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
class MobileNetV2(nn.Module):
    __factory = {
        320: torchvision.models.mobilenet_v2,
     }
    def __init__(self, depth = 320,num_classes=1000, width_mult=1.0, pretrained=True,matconvnet = None,cut_at_pooling=False):
        super(MobileNetV2, self).__init__()
        self.pretrained = pretrained
        self.matconvnet = matconvnet
        # setting of inverted residual blocks
        mobilenet_v2 = MobileNetV2.__factory[depth](pretrained=pretrained)
        self.feature_dim = 320
        self.cut_at_pooling = cut_at_pooling
        layers = list(mobilenet_v2.features.children())[:-1]
        #layers.append(CBAMBlock())
        self.features = nn.Sequential(*layers) # capture only feature part and remove last relu and maxpool
        self.gap = nn.AdaptiveMaxPool2d(1)
        # building last several layers
        #output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        #self.conv = conv_1x1_bn(input_channel, output_channel)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Linear(output_channel, num_classes)
        
        self._init_params()

    def forward(self, x):
        x = self.features(x)
        if self.cut_at_pooling:
            return x
        pool_x = self.gap(x)
        pool_x = pool_x.view(pool_x.size(0), -1)

        return pool_x, x
       # x = self.conv(x)
      #  x = self.avgpool(x)
       # x = x.view(x.size(0), -1)
       # x = self.classifier(x)
    def _init_params(self):
        # optional load pretrained weights from matconvnet
        if self.matconvnet is not None:
            self.features.load_state_dict(torch.load(models_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')))
            self.pretrained = True
        
        #for m in self.modules():
         #   if isinstance(m, nn.Conv2d):
         #       n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
         #       m.weight.data.normal_(0, math.sqrt(2. / n))
         #       if m.bias is not None:
         #           m.bias.data.zero_()
         #   elif isinstance(m, nn.BatchNorm2d):
         #       m.weight.data.fill_(1)
         #       m.bias.data.zero_()
         #   elif isinstance(m, nn.Linear):
         #       m.weight.data.normal_(0, 0.01)
         #       m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(depth = 320,**kwargs)
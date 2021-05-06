import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['mlp_5layer']


class MLP(nn.Module):
    def __init__(self, num_classes=128, in_channels=[3], hiddens=[64, 64], batchnorm=nn.BatchNorm1d, **kwargs):
        super(MLP, self).__init__()
        self.hiddens = hiddens
        self.body = []
        if isinstance(in_channels, (tuple, list)):
            self.splits = torch.tensor([0] + list(in_channels)).cumsum(dim=0)
            for idx, ch in enumerate(in_channels):
                setattr(self, 'fc%d' %(idx+1), nn.Linear(ch, 64))
            hiddens = [64 * len(in_channels)] + hiddens
        else:
            self.splits = None
            setattr(self, 'fc1', self._make_layer(in_channels, 64, batchnorm, activation=True))
            hiddens = [64] + hiddens
        for i in range(len(hiddens)-2):
            self.body.append(self._make_layer(hiddens[i], hiddens[i+1], batchnorm, activation=True))
        self.body = nn.Sequential(*self.body)
        self.last_fc = nn.Linear(hiddens[-2], hiddens[-1])
        self.fc = nn.Linear(hiddens[-1]//4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (batchnorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channel, out_channel, batchnorm, activation=True):
        layers = []
        layers.append(nn.Linear(in_channel, out_channel, bias=(batchnorm != None)))
        if batchnorm is not None:
            layers.append(batchnorm(out_channel))
        if activation is not None:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, no_fc=False):
        if self.splits is not None:
            out = torch.cat([self.__getattr__(f'fc{i+1}')(x[:,self.splits[i]:self.splits[i+1]]) for i in range(len(self.splits)-1)], axis=1)
        else:
            out = self.fc1(x)
        out = self.body(out)
        out = self.last_fc(out)
        out = out.view(-1, self.hiddens[-1]//4, 4).max(2)[0]
        if not no_fc:
            out = self.fc(out)
        return out


def mlp_5layer(**kwargs):
    width = kwargs['width'] if 'width' in kwargs else 64
    return MLP(hiddens=[32 * width, 32 * width, 64 * width, 64 * width, 128 * width], **kwargs)


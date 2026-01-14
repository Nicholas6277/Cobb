import torch
import torch.nn as nn
from wtconv.wtconv2d import WTConv2d


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SEResnet(nn.Module):
    def __init__(self, channel_in, channel_out, stride, padding=1, r=16):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(channel_out),
            SE_Block(channel_out, r)
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.sequence(x)
        out += x
        out = self.ReLU(out)
        return out

class conv3x3_bn_relu(nn.Module):
    def __init__(self, channel_in, channel_out, stride, padding=1):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sequence(x)

class down(nn.Module):
    def __init__(self, channel_in, channel_out, stride, padding=1, pooling=True, r=16):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3_bn_relu(channel_in, channel_out, stride, padding),
            SEResnet(channel_out, channel_out, stride, padding, r),
            SEResnet(channel_out, channel_out, stride, padding, r),
        )
        self.max_pooling = nn.MaxPool2d(2) if pooling else nn.Identity()

    def forward(self, x):
        return self.conv(self.max_pooling(x))

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class up(nn.Module):
    def __init__(self, channel_in, channel_out, stride=1, padding=1, bilinear=True):  # bilinear：双线性插值进行上采样
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # align_corners=True：输入图像的角点与输出图像的角点完全对齐
            self.conv = nn.Sequential(
                conv3x3_bn_relu(channel_in + channel_in // 2, channel_out, stride, padding),
                conv3x3_bn_relu(channel_out, channel_out, stride, padding),
            )
        else:
            self.up = nn.ConvTranspose2d(channel_in, channel_in // 2, kernel_size=2, stride=stride)
            self.conv = nn.Sequential(
                conv3x3_bn_relu(channel_in, channel_out, stride, padding),
                conv3x3_bn_relu(channel_out, channel_out, stride, padding),
            )
        self.AG = Attention_block(channel_in, channel_out, channel_out)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        att = self.AG(x2, x1)
        return self.conv(torch.cat([att, x2], dim=1))

class myModel(nn.Module):
    def __init__(self, n_channel=1, n_class=1, bilinear=True):
        super().__init__()
        self.down1 = down(n_channel, 12, 1, pooling=False)
        self.down2 = down(12, 24, 1)
        self.down3 = down(24, 48, 1)
        self.down4 = down(48, 96, 1)
        self.down5 = down(96, 192, 1)
        self.up1 = up(96 * 2, 96, 1, bilinear=bilinear)
        self.up2 = up(48 * 2, 48, 1, bilinear=bilinear)
        self.up3 = up(24 * 2, 24, 1, bilinear=bilinear)
        self.up4 = up(12 * 2, 12, 1, bilinear=bilinear)
        self.final = nn.Conv2d(12, n_class, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        return self.final(x)

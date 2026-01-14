import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, inplanes, kernels, stride, padding=1, r=16):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(inplanes, kernels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(kernels),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernels, kernels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(kernels),
            SE_Block(kernels, r)
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.sequence(x)
        out += x
        out = self.ReLU(out)
        return out

class conv3x3_bn_relu(nn.Module):
    def __init__(self, inplanes, kernels, stride, padding=1):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(inplanes, kernels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(kernels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sequence(x)

class down(nn.Module):
    def __init__(self, inplanes, kernels, stride, padding=1, pooling=True, r=16):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3_bn_relu(inplanes, kernels, stride, padding),
            SEResnet(kernels, kernels, stride, padding, r),
            SEResnet(kernels, kernels, stride, padding, r),
        )
        self.max_pooling = nn.MaxPool2d(2) if pooling else nn.Identity()

    def forward(self, x):
        return self.conv(self.max_pooling(x))


class MSR(nn.Module):
    def __init__(self, scales=[15, 80, 250]):
        super(MSR, self).__init__()
        self.scales = scales

    def forward(self, x):
        msr_result = []
        for scale in self.scales:
            kernel_size = int(4 * scale + 1)
            padding = kernel_size // 2
            sigma = scale
            # 使用深度卷积代替高斯滤波
            smoothed = self.depthwise_conv(x, kernel_size, sigma)
            msr_result.append(torch.log1p(x) - torch.log1p(smoothed))
        msr_fused = torch.mean(torch.stack(msr_result), dim=0)
        return msr_fused

    def depthwise_conv(self, x, kernel_size, sigma):
        # 创建一个卷积核，尺寸为 kernel_size，并且具有高斯形状
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        grid = coords.repeat(kernel_size, 1)
        gaussian = torch.exp(-(grid ** 2 + grid.T ** 2) / (2 * sigma ** 2))
        gaussian /= gaussian.sum()

        # 将卷积核调整为适应深度卷积
        gaussian = gaussian.view(1, 1, kernel_size, kernel_size)
        gaussian = gaussian.repeat(x.shape[1], 1, 1, 1).to(x.device)  # 扩展到每个输入通道

        # 使用深度卷积操作
        smoothed = F.conv2d(x, gaussian, padding=kernel_size // 2, groups=x.shape[1])
        return smoothed

class AttentionWithMSR(nn.Module):
    def __init__(self, F_g, F_l, F_int, scales=[15, 80, 250]):
        super().__init__()
        self.msr = MSR(scales)

        # Feature transformation for gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Feature transformation for input feature
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Generate attention weights
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Apply MSR enhancement to the input feature map
        msr_enhanced = self.msr(x)

        # Compute attention weights
        psi = self.relu(self.W_g(g) + self.W_x(msr_enhanced))
        psi = self.psi(psi)

        # Apply attention to the input feature map
        return x * psi

class up(nn.Module):
  def __init__(self, inplanes, kernels, stride = 1, padding = 1, bilinear = True):
    super().__init__()
    if bilinear :
      self.up = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners = True)
      self.conv = nn.Sequential(
        conv3x3_bn_relu(inplanes + inplanes // 2, kernels, stride, padding),
        conv3x3_bn_relu(kernels, kernels, stride, padding),
      )
    else:
      self.up = nn.ConvTranspose2d(inplanes, inplanes//2, kerenls_size = 2, stride = stride)

      self.conv = nn.Sequential(
        conv3x3_bn_relu(inplanes, kernels, stride, padding),
        conv3x3_bn_relu(kernels, kernels, stride, padding),
      )
    self.AG = AttentionWithMSR(inplanes, kernels, kernels)


  def forward(self, x1, x2):
    x2 = self.up(x2)
    att = self.AG(x2, x1)
    return self.conv(torch.cat([att, x2], dim = 1))

class myModel_2(nn.Module):
  def __init__(self, n_channel = 1, n_class = 1, bilinear = True):
    super().__init__()
    self.down1 = down(n_channel, 12, 1, pooling = False)
    self.down2 = down(12, 24, 1)
    self.down3 = down(24, 48, 1)
    self.down4 = down(48, 96, 1)
    self.down5 = down(96, 192, 1)
    self.up1 = up(96*2, 96, 1, bilinear = bilinear)
    self.up2 = up(48*2, 48, 1, bilinear = bilinear)
    self.up3 = up(24*2, 24, 1, bilinear = bilinear)
    self.up4 = up(12*2, 12, 1, bilinear = bilinear)
    self.out = nn.Conv2d(12, n_class, 3, 1, padding = 1)
  def forward(self, x):
    x1 = self.down1(x)
    x2 = self.down2(x1)
    x3 = self.down3(x2) #channel:48
    x4 = self.down4(x3) #channel:96   shape: torch.Size([1, 96, 152, 64])
    x5 = self.down5(x4) #channel:192   shape: torch.Size([1, 192, 76, 32])

    x = self.up1(x4, x5) #channel:96
    x = self.up2(x3, x) #channel:48
    x = self.up3(x2, x) #channel:24
    x = self.up4(x1, x) #channek:12
    logit = self.out(x)
    return logit
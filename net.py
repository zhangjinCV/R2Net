import paddle
from paddle import nn
from paddle.nn import functional as F
from paddleseg.cvlibs.param_init import kaiming_normal_init, constant_init


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0.0)

        elif isinstance(m, nn.Conv1D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0.0)

        elif isinstance(m, (nn.BatchNorm2D, nn.InstanceNorm2D)):
            constant_init(m.weight, value=1.0)
            if m.bias is not None:
                constant_init(m.bias, value=0.0)
        elif isinstance(m, nn.Linear):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0.0)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2D):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2D):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.MaxPool2D):
            pass
        elif isinstance(m, nn.Hardswish):
            pass
        else:
            m.init_weight()


class CA(nn.Layer):
    def __init__(self, in_ch, reduction=32):
        super(CA, self).__init__()
        self.cv1 = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_ch, in_ch // reduction, 1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(in_ch // reduction, in_ch, 1, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.cv1(x)
        x = x * identity
        return x

    def init_weight(self):
        weight_init(self)


class ResNet(paddle.nn.Layer):
    def __init__(self):
        super(ResNet, self).__init__()
        r50 = paddle.vision.models.resnet50(True)
        self.cv1 = r50.conv1
        self.bn1 = r50.bn1
        self.maxp = r50.maxpool
        self.relu = r50.relu
        self.layer1 = r50.layer1
        self.layer2 = r50.layer2
        self.layer3 = r50.layer3
        self.layer4 = r50.layer4

    def forward(self, x):
        x1 = self.maxp(self.relu(self.bn1(self.cv1(x))))
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5

    def init_weight(self):
        print("bkbone finished!")


class RPM(nn.Layer):
    def __init__(self):
        super(RPM, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 3, dilation=3),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 5, dilation=5),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 7, dilation=7),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv4 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 9, dilation=9),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv5 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        self.se = CA(64 * 5, 5)
        self.cv6 = nn.Sequential(
            nn.Conv2D(64 * 5, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
        )

    def forward(self, x, y):
        x1 = self.cv1(y)
        x2 = self.cv2(y)
        x3 = self.cv3(y)
        x4 = self.cv4(y)
        x5 = self.cv5(y)
        xs = paddle.concat([x1, x2, x3, x4, x5], 1)
        xs = self.se(xs)
        x = F.relu(x + self.cv6(xs))
        return x

    def init_weight(self):
        weight_init(self)


class rpms(nn.Layer):
    def __init__(self):
        super(rpms, self).__init__()
        self.rpm2, self.rpm3, self.rpm4, self.rpm5 = RPM(), \
                                                 RPM(), \
                                                 RPM(), \
                                                 RPM()

    def forward(self, x2, x3, x4, x5):
        out2, out3, out4, out5 = self.rpm2(x2, x2), \
                                 self.rpm3(x3, x3), self.rpm4(x4, x4), \
                                 self.rpm5(x5, x5)

        return out2, out3, out4, out5

    def init_weight(self):
        weight_init(self)


class Squeeze(nn.Layer):
    def __init__(self):
        super(Squeeze, self).__init__()
        self.cv2 = nn.Sequential(nn.Conv2D(256, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2D(64), nn.ReLU())
        self.cv3 = nn.Sequential(nn.Conv2D(512, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2D(64), nn.ReLU())
        self.cv4 = nn.Sequential(nn.Conv2D(1024, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2D(64), nn.ReLU())
        self.cv5 = nn.Sequential(nn.Conv2D(2048, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2D(64), nn.ReLU())

    def forward(self, inp2, inp3, inp4, inp5):
        inp2 = self.cv2(inp2)
        inp3 = self.cv3(inp3)
        inp4 = self.cv4(inp4)
        inp5 = self.cv5(inp5)
        return inp2, inp3, inp4, inp5

    def init_weight(self):
        weight_init(self)


class RFM(nn.Layer):
    def __init__(self):
        super(RFM, self).__init__()
        self.cv1 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2D(64)

        self.cv2 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2D(64)

        self.cv3 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2D(64)

        self.cv4 = nn.Conv2D(64 * 3, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2D(64)

    def forward(self, left, mid, right):
        left_  = F.relu(self.bn3(self.cv3(left)))
        right = F.interpolate(right, size=left.shape[2:], mode='bilinear')
        right = F.relu(self.bn1(self.cv1(right)))
        mid = F.interpolate(mid, size=left.shape[2:], mode='bilinear')
        mid = F.relu(self.bn2(self.cv2(mid)))

        lm = left_ * mid
        lr = left_ * right
        mr = mid * right

        cat = paddle.concat([lm, lr, mr], 1)
        cat = self.bn4(self.cv4(cat))
        cat = F.relu(left + cat)
        return cat

    def init_weight(self):
        weight_init(self)


class rfms(nn.Layer):
    def __init__(self):
        super(rfms, self).__init__()
        self.rf1 = RFM()
        self.rf2 = RFM()
        self.rf3 = RFM()
        self.fo1 = FOM()
        self.fo2 = FOM()
        self.fo3 = FOM()
        self.fo4 = FOM()

    def forward(self, out2, out3, out4, out5):
        out5 = self.fo4(out5)
        out4 = self.fo1(self.rf1(out4, out5, out5))
        out3 = self.fo2(self.rf2(out3, out4, out5))
        out2 = self.fo3(self.rf3(out2, out3, out5))
        return out2, out3, out4, out5

    def init_weight(self):
        weight_init(self)


class FOM(nn.Layer):
    def __init__(self):
        super(FOM, self).__init__()
        self.cv1 = nn.Sequential(nn.Conv2D(64, 64 * 2, 3, 1, 1), nn.BatchNorm2D(64 * 2), nn.ReLU())
        self.cv2 = nn.Sequential(nn.Conv2D(64, 64, 3, 1, 1), nn.BatchNorm2D(64))

    def forward(self, x):
        y = self.cv1(x)
        w, b = paddle.split(y, 2, 1)
        x = F.relu(x * w + self.cv2(b))
        return x

    def init_weight(self):
        weight_init(self)


class R2Net(nn.Layer):
    def __init__(self, cfg):
        super(R2Net, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet()
        self.squeeze = Squeeze()
        self.rpms = rpms()
        self.rfm = rfms()

        self.linear1 = nn.Conv2D(64, 1, 3, 1, 1)
        self.linear2 = nn.Conv2D(64, 1, 3, 1, 1)
        self.linear3 = nn.Conv2D(64, 1, 3, 1, 1)
        self.linear4 = nn.Conv2D(64, 1, 3, 1, 1)

        for p in self.bkbone.parameters():
            p.optimize_attr['learning_rate'] /= 10.0

        self.init_weight()

    def forward(self, x):
        out1, out2, out3, out4, out5 = self.bkbone(x)
        out2, out3, out4, out5 = self.squeeze(out2, out3, out4, out5)
        out2, out3, out4, out5 = self.rpms(out2, out3, out4, out5)
        out2, out3, out4, out5 = self.rfm(out2, out3, out4, out5)

        out2 = F.interpolate(self.linear1(out2), size=x.shape[2:], mode='bilinear')
        out3 = F.interpolate(self.linear2(out3), size=x.shape[2:], mode='bilinear')
        out4 = F.interpolate(self.linear3(out4), size=x.shape[2:], mode='bilinear')
        out5 = F.interpolate(self.linear4(out5), size=x.shape[2:], mode='bilinear')
        return out2, out3, out4, out5

    def init_weight(self):
        weight_init(self)


if __name__ == '__main__':
    import pandas as pd

    f4 = R2Net(1)
    x = paddle.ones((2, 3, 224, 224))
    f4.load_dict(paddle.load(r'R2Net.pdparams'))
    out2 = f4(x)
    # print(out1[0][0][2])
    total_params = sum(p.numel() for p in f4.parameters())
    print('total params : ', total_params)
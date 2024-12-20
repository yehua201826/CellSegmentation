import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

__all__ = ["MILresnet18", "MILresnet34", "MILresnet50"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MILResNet(nn.Module):

    def __init__(self, encoder, block, layers, num_classes=1000, expansion=1):

        self.encoder_name = encoder
        self.mode = None
        self.encoder_prefix = (
            "conv1",
            "bn1",
            "relu",
            "layer1",
            "layer2",
            "layer3",
            "layer4"
        )
        self.image_module_prefix = (
            "fc_image_cls",
            "fc_image_reg"
        )
        self.tile_module_prefix = (
            "fc_tile",
        )
        self.seg_module_prefix = (
            "upconv",
            "seg_out_conv"
        )

        self.inplanes = 64
        super(MILResNet, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # heads beneath the encoder
        def init_tile_modules():
            self.avgpool_tile = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool_tile = nn.AdaptiveMaxPool2d((1, 1))
            self.fc_tile = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * block.expansion, num_classes)
            )

        def init_image_modules(map_size):
            self.avgpool_image = nn.AdaptiveAvgPool2d((map_size, map_size))
            self.maxpool_image = nn.AdaptiveMaxPool2d((map_size, map_size))
            self.fc_image_cls = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(512 * map_size * map_size * block.expansion),
                nn.Dropout(p=0.25),
                nn.ReLU(inplace=True),
                nn.Linear(512 * map_size * map_size * block.expansion, 64),
                nn.BatchNorm1d(64),
                nn.Dropout(),
                nn.Linear(64, 7)
            )
            self.fc_image_reg = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(512 * map_size * map_size * block.expansion),
                nn.Dropout(p=0.25),
                nn.ReLU(inplace=True),
                nn.Linear(512 * map_size * map_size * block.expansion, 64),
                nn.BatchNorm1d(64),
                nn.Dropout(),
                nn.Linear(64, 1),
                nn.ReLU(inplace=True)
            )

        def init_seg_modules():
            # upsample convolution layers
            self.upconv1 = self.upsample_conv(512 * expansion, 256 * expansion)
            self.upconv2 = self.upsample_conv(512 * expansion, 256 * expansion)
            self.upconv3 = self.upsample_conv(256 * expansion, 128 * expansion)
            self.upconv4 = self.upsample_conv(256 * expansion, 128 * expansion)
            self.upconv5 = self.upsample_conv(128 * expansion, 64 * expansion)
            self.upconv6 = self.upsample_conv(128 * expansion, 64 * expansion)
            self.upconv7 = self.upsample_conv(64 * expansion, 64 if expansion == 1 else 32 * expansion)
            self.upconv8 = self.upsample_conv(64 if expansion == 1 else 32 * expansion, 64)
            self.seg_out_conv = nn.Conv2d(64, 2, kernel_size=1)

        init_tile_modules()
        init_image_modules(map_size=1)
        init_seg_modules()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def upsample_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def set_encoder_grads(self, requires_grad):

        self.conv1.requires_grad_(requires_grad)
        self.bn1.requires_grad_(requires_grad)
        self.relu.requires_grad_(requires_grad)
        self.maxpool.requires_grad_(requires_grad)
        self.layer1.requires_grad_(requires_grad)
        self.layer2.requires_grad_(requires_grad)
        self.layer3.requires_grad_(requires_grad)
        self.layer4.requires_grad_(requires_grad)

    def set_tile_module_grads(self, requires_grad):

        self.avgpool_tile.requires_grad_(requires_grad)
        self.maxpool_tile.requires_grad_(requires_grad)
        self.fc_tile.requires_grad_(requires_grad)

    def set_image_module_grads(self, requires_grad):

        self.avgpool_image.requires_grad_(requires_grad)
        self.maxpool_image.requires_grad_(requires_grad)
        self.fc_image_cls.requires_grad_(requires_grad)
        self.fc_image_reg.requires_grad_(requires_grad)

    def set_seg_module_grads(self, requires_grad):

        self.upconv1.requires_grad_(requires_grad)
        self.upconv2.requires_grad_(requires_grad)
        self.upconv3.requires_grad_(requires_grad)
        self.upconv4.requires_grad_(requires_grad)
        self.seg_out_conv.requires_grad_(requires_grad)

    def resnet_forward(self, x: torch.Tensor, return_intermediate: bool = False):

        x = self.conv1(x)       # x_tile: [nk,  64, 16, 16] x_image: [n,   64, 150, 150]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # x_tile: [nk,  64,  8,  8] x_image: [n,   64,  75,  75]
        x1 = self.layer1(x)     # x_tile: [nk,  64,  8,  8] x_image: [n,  256,  75,  75]
        x2 = self.layer2(x1)    # x_tile: [nk, 128,  4,  4] x_image: [n,  512,  38,  38]
        x3 = self.layer3(x2)    # x_tile: [nk, 256,  2,  2] x_image: [n, 1024,  19,  19]
        x4 = self.layer4(x3)    # x_tile: [nk, 512,  1,  1] x_image: [n, 2048,  10,  10]

        if return_intermediate:
            return x4, x3, x2, x1
        else:
            return x4

    def forward(self, x: torch.Tensor, freeze_bn=False):  # x_tile: [nk, 3, 32, 32] x_image: [n, 3, 299, 299]

        # Set freeze_bn=True in tile training mode to freeze E(x) & Var(x) in BatchNorm2D(x).
        # Otherwise, assessment results will decay as tile training propels.
        if self.mode == "tile" and freeze_bn:
            # TODO: iterate named_modules() and use bn.eval()
            self.eval()
            x4 = self.resnet_forward(x, False)
            self.train()
        elif self.mode == "segment":
            x4, x3, x2, x1 = self.resnet_forward(x, True)
        else:
            x4 = self.resnet_forward(x, False)

        if self.mode == "tile":

            x = self.avgpool_tile(x4) + self.maxpool_tile(x4) # x: [nk, 512, 1, 1]
            x = self.fc_tile(x)  # x: [nk, 512]

            return x

        elif self.mode == "image":

            # image_cls & image_reg
            out = self.avgpool_image(x4) + self.maxpool_image(x4)  # [n, 2048, ?, ?]
            out_cls = self.fc_image_cls(out)  # [n, 7]
            out_reg = self.fc_image_reg(out)  # [n, 1]

            return out_cls, out_reg

        elif self.mode == "segment":
            # lysto
            out_seg = F.interpolate(x4.clone(), size=19, mode="bilinear", align_corners=True)   # out_seg: [n, 2048, 19, 19]
            out_seg = self.upconv1(out_seg)                                                     # [n, 1024, 19, 19]
            out_seg = torch.cat([out_seg, x3], dim=1)                                           # concat: [n, 2048, 19, 19]
            out_seg = self.upconv2(out_seg)                                                     # [n, 1024, 19, 19]

            out_seg = F.interpolate(out_seg, size=38, mode="bilinear", align_corners=True)      # [n, 1024, 38, 38]
            out_seg = self.upconv3(out_seg)                                                     # [n, 512, 38, 38]
            out_seg = torch.cat([out_seg, x2], dim=1)                                           # concat: [n, 1024, 38, 38]
            out_seg = self.upconv4(out_seg)                                                     # [n, 512, 38, 38]

            out_seg = F.interpolate(out_seg, size=75, mode="bilinear", align_corners=True)      # [n, 512, 75, 75]
            out_seg = self.upconv5(out_seg)                                                     # [n, 256, 75, 75]
            out_seg = torch.cat([out_seg, x1], dim=1)                                           # concat: [n, 512, 75, 75]
            out_seg = self.upconv6(out_seg)                                                     # [n, 256, 75, 75]

            out_seg = F.interpolate(out_seg, size=150, mode="bilinear", align_corners=True)     # [n, 256, 150, 150]
            out_seg = self.upconv7(out_seg)                                                     # [n, 128, 150, 150]
            out_seg = self.upconv8(out_seg)                                                     # [n, 64, 150, 150]
            out_seg = F.interpolate(out_seg, size=299, mode="bilinear", align_corners=True)     # [n, 64, 299, 299]
            out_seg = self.seg_out_conv(out_seg)                                                # [n, 1, 299, 299]

            # nuclick
            # out_seg = F.interpolate(x4.clone(), size=16, mode="bilinear",
            #                         align_corners=True)  # out_seg: [n, 2048, 19, 19]
            # out_seg = self.upconv1(out_seg)  # [n, 1024, 19, 19]
            # out_seg = torch.cat([out_seg, x3], dim=1)  # concat: [n, 2048, 19, 19]
            # out_seg = self.upconv2(out_seg)  # [n, 1024, 19, 19]
            #
            # out_seg = F.interpolate(out_seg, size=32, mode="bilinear", align_corners=True)  # [n, 1024, 38, 38]
            # out_seg = self.upconv3(out_seg)  # [n, 512, 38, 38]
            # out_seg = torch.cat([out_seg, x2], dim=1)  # concat: [n, 1024, 38, 38]
            # out_seg = self.upconv4(out_seg)  # [n, 512, 38, 38]
            #
            # out_seg = F.interpolate(out_seg, size=64, mode="bilinear", align_corners=True)  # [n, 512, 75, 75]
            # out_seg = self.upconv5(out_seg)  # [n, 256, 75, 75]
            # out_seg = torch.cat([out_seg, x1], dim=1)  # concat: [n, 512, 75, 75]
            # out_seg = self.upconv6(out_seg)  # [n, 256, 75, 75]
            #
            # out_seg = F.interpolate(out_seg, size=128, mode="bilinear", align_corners=True)  # [n, 256, 150, 150]
            # out_seg = self.upconv7(out_seg)  # [n, 128, 150, 150]
            # out_seg = self.upconv8(out_seg)  # [n, 64, 150, 150]
            # out_seg = F.interpolate(out_seg, size=256, mode="bilinear", align_corners=True)  # [n, 64, 299, 299]
            # out_seg = self.seg_out_conv(out_seg)

            return out_seg

        else:
            raise Exception("Something wrong in setmode.")

    def setmode(self, mode):
        """
        mode "image":   pt.1 (whole image mode), pooled feature -> image classification & regression
        mode "tile":    pt.2 (instance mode), pooled feature -> tile classification
        mode "segment": pt.3 (segmentation mode), pooled feature -> expanding path -> output map
        """

        if mode == "tile":
            self.set_encoder_grads(False)
            self.set_tile_module_grads(True)
            self.set_image_module_grads(False)
            self.set_seg_module_grads(False)
        elif mode == "image":
            self.set_encoder_grads(True)
            self.set_tile_module_grads(False)
            self.set_image_module_grads(True)
            self.set_seg_module_grads(False)
        elif mode == "segment":
            self.set_encoder_grads(False)
            self.set_tile_module_grads(False)
            self.set_image_module_grads(False)
            self.set_seg_module_grads(True)
        else:
            raise Exception("Invalid mode: {}.".format(mode))

        self.mode = mode


def MILresnet18(pretrained=False, **kwargs):

    model = MILResNet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    # change num of cell classes from 1000 to 2 here to make it compatible with pretrained files
    model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
    return model


def MILresnet34(pretrained=False, **kwargs):

    model = MILResNet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
    return model


def MILresnet50(pretrained=False, **kwargs):

    model = MILResNet('resnet50', Bottleneck, [3, 4, 6, 3], expansion=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    model.fc_tile[1] = nn.Linear(model.fc_tile[1].in_features, 2)
    return model

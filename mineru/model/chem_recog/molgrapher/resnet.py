"""ResNet (18/50) replica of ``mol_depict.utils.utils_graph_classifier``.

Kept structurally identical to the upstream definition because the official
MolGrapher checkpoints are keyed on it:

* ``conv1`` 7x7 conv with configurable stride, then BN/ReLU/3x3 maxpool
* ``layer1..4`` of ``BasicBlock`` (resnet18) or ``Bottleneck`` (resnet50)
* **only the first block of a layer receives the layer's stride/dilation**
  (upstream behaviour, weights were trained with it - do not "fix")
* ``avgpool`` + ``fc`` are always built (they are part of the checkpoint for
  the graph classifier, even though only ``layer4`` is used).

No torchvision / model zoo dependency: weights always come from the
MolGrapher checkpoints.
"""
import math
from collections import OrderedDict

import torch.nn as nn

__all__ = ["BasicBlock", "Bottleneck", "ResNet", "resnet18", "resnet50"]


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1,
                 use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.relu(out)


class ResNet(nn.Module):
    """ResNet with partial-output support (``layer4`` is what MolGrapher uses)."""

    def __init__(self, block, layers, output_layers, conv1_stride=2, num_classes=1000,
                 inplanes=64, dilation_factor=1):
        self.inplanes = inplanes
        super().__init__()
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=conv1_stride,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(block, inplanes, layers[0],
                                       dilation=max(dilation_factor // 8, 1))
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1],
                                       stride=stride[0],
                                       dilation=max(dilation_factor // 4, 1))
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2],
                                       stride=stride[1],
                                       dilation=max(dilation_factor // 2, 1))
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3],
                                       stride=stride[2], dilation=dilation_factor)

        out_feature_strides = {
            "conv1": 4,
            "layer1": 4,
            "layer2": 4 * stride[0],
            "layer3": 4 * stride[0] * stride[1],
            "layer4": 4 * stride[0] * stride[1] * stride[2],
        }
        if isinstance(self.layer1[0], BasicBlock):
            out_feature_channels = {"conv1": inplanes, "layer1": inplanes,
                                    "layer2": inplanes * 2, "layer3": inplanes * 4,
                                    "layer4": inplanes * 8}
        else:
            base = 4 * inplanes
            out_feature_channels = {"conv1": inplanes, "layer1": base,
                                    "layer2": base * 2, "layer3": base * 4,
                                    "layer4": base * 8}
        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        return self._out_feature_strides if layer is None else self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        return self._out_feature_channels if layer is None else self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # NOTE: upstream passes neither stride nor dilation for blocks > 0
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        outputs = OrderedDict()
        if output_layers is None:
            output_layers = self.output_layers

        x = self.relu(self.bn1(self.conv1(x)))
        if self._add_output_and_check("conv1", x, outputs, output_layers):
            return outputs
        x = self.maxpool(x)
        x = self.layer1(x)
        if self._add_output_and_check("layer1", x, outputs, output_layers):
            return outputs
        x = self.layer2(x)
        if self._add_output_and_check("layer2", x, outputs, output_layers):
            return outputs
        x = self.layer3(x)
        if self._add_output_and_check("layer3", x, outputs, output_layers):
            return outputs
        x = self.layer4(x)
        if self._add_output_and_check("layer4", x, outputs, output_layers):
            return outputs
        x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), -1))
        if self._add_output_and_check("fc", x, outputs, output_layers):
            return outputs
        if len(output_layers) == 1 and output_layers[0] == "default":
            return x
        raise ValueError("output_layer is wrong.")


def _check_layers(output_layers):
    if output_layers is None:
        return ["default"]
    for l in output_layers:
        if l not in ("conv1", "layer1", "layer2", "layer3", "layer4", "fc"):
            raise ValueError(f"Unknown layer: {l}")
    return output_layers


def resnet18(output_layers=None, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], _check_layers(output_layers), **kwargs)


def resnet50(output_layers=None, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], _check_layers(output_layers), **kwargs)

import os
import sys
import torch
import torch.nn as nn
import math
# sys.path.insert(0, "../../SparseConvNet")
# import sparseconvnet as scn
from lib.nn import SynchronizedBatchNorm2d

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

import time

import spconv.pytorch as spconv


__all__ = ['ResNet', 'resnet50', 'resnet101']


model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}

class roundGrad(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input


# class AddSparseDense(nn.Sequential):
#     def __init__(self, *args):
#         nn.Sequential.__init__(self, *args)
#
#     def forward(self, input):
#         a = input[0]
#         b = input[1]
#         output = scn.SparseConvNetTensor()
#         output.metadata = a.metadata
#         output.spatial_size = a.spatial_size
#         axyz = a.get_spatial_locations()
#         y = axyz[:,0]
#         x = axyz[:,1]
#         z = axyz[:,2]
#
#
#         output.features = a.features + b[z,:,y,x]
#         return output
#
#     def input_spatial_size(self,out_size):
#         return out_size


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# def conv3x3_sparse(in_planes, out_planes, stride=1):
#     "3x3 sparse convolution"
#     if stride == 1:
#         return scn.SubmanifoldConvolution(2, in_planes, out_planes, 3, False)
#     else:
#         return scn.Convolution(2, in_planes, out_planes, 3, stride, False)

# def deconv3x3_sparse(in_planes, out_planes, stride=1):
#     if stride == 1:
#         return scn.SubmanifoldConvolution(2, in_planes, out_planes, 3, False)
#     else:
#         return scn.Deconvolution(2, in_planes, out_planes, 3, stride, False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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



# class BasicBlockSparse(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlockSparse, self).__init__()
#         self.conv1 = conv3x3_sparse(inplanes, planes, stride)
#         self.bn1 = scn.BatchNormReLU(planes)
#         self.relu = scn.ReLU()
#         self.conv2 = conv3x3_sparse(planes, planes)
#         self.bn2 = scn.BatchNormReLU(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = SynchronizedBatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


# class TransBasicBlockSparse(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
#         super(TransBasicBlockSparse, self).__init__()
#         self.conv1 = deconv3x3_sparse(inplanes, inplanes)
#         self.bn1 = scn.BatchNormReLU(inplanes)
#         self.relu = scn.ReLU()
#         self.conv2 = deconv3x3_sparse(inplanes, planes, stride=stride)
#         self.bn2 = scn.BatchNormalization(planes)
#         self.add = scn.AddTable()
#         self.stride = stride
#         self.upsample = upsample
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.upsample is not None:
#             residual = self.upsample(x)
#
#         out = self.add([out,residual])
#         out = self.relu(out)
#
#         return out

# class TransBasicBlockSparse(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
#         super(TransBasicBlockSparse, self).__init__()
#         self.conv1 = conv3x3_sparse(inplanes, inplanes)
#         self.bn1 = scn.BatchNormReLU(inplanes)
#         self.relu = scn.ReLU()
#         # if upsample is not None and stride != 1:
#         #     self.conv2 = scn.Sequential(
#         #         scn.SparseToDense(2,inplanes),
#         #         nn.ConvTranspose2d(inplanes, planes,
#         #                           kernel_size=2, stride=stride, padding=0,
#         #                           output_padding=0, bias=False),
#         #         scn.DenseToSparse(2)
#         #     )
#         # else:
#         #     self.conv2 = conv3x3_sparse(inplanes, planes, stride)
#         self.conv2 = deconv3x3_sparse(inplanes, planes, stride)
#         self.bn2 = scn.BatchNormalization(planes)
#         self.add = scn.AddTable()
#         self.upsample = upsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.upsample is not None:
#             residual = self.upsample(x)
#
#         out = self.add([out,residual])
#         out = self.relu(out)
#
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = SynchronizedBatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = SynchronizedBatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = SynchronizedBatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        hiddenplanes = inplanes * 4
        self.conv1 = nn.Conv2d(inplanes, hiddenplanes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(hiddenplanes)
        self.conv2 = nn.Conv2d(hiddenplanes, hiddenplanes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.conv3 = nn.Conv2d(hiddenplanes, planes, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        self.stride = stride

        self.use_res_connect = self.stride==1 and inplanes==planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        #
        # out += residual
        if self.use_res_connect:
            out += residual

        out = self.relu(out)

        return out


class TransBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes * 4, inplanes, kernel_size=1, bias=False)
        self.bn1 = SynchronizedBatchNorm2d(inplanes)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, inplanes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, inplanes, stride)
        self.bn2 = SynchronizedBatchNorm2d(inplanes)
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn3 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out



# class TransBottleneckSparse(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
#         super(TransBottleneckSparse, self).__init__()
#         self.conv1 = scn.NetworkInNetwork(inplanes * 4, inplanes, False)
#         self.bn1 = scn.BatchNormReLU(inplanes)
#         if upsample is not None and stride != 1:
#             self.conv2 = scn.Sequential(
#                 scn.SparseToDense(2,inplanes),
#                 nn.ConvTranspose2d(inplanes, inplanes,
#                                   kernel_size=2, stride=stride, padding=0,
#                                   output_padding=0, bias=False),
#                 scn.DenseToSparse(2)
#             )
#         else:
#             self.conv2 = conv3x3_sparse(inplanes, inplanes, stride)
#         self.bn2 = scn.BatchNormReLU(inplanes)
#         self.conv3 = scn.NetworkInNetwork(inplanes, planes, False)
#         self.bn3 = scn.BatchNormalization(planes)
#         self.relu = scn.ReLU()
#         self.add = scn.AddTable()
#         self.upsample = upsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.upsample is not None:
#             residual = self.upsample(x)
#
#         out = self.add([out,residual])
#         out = self.relu(out)
#
#         return out

# class TransBottleneckSparse(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
#         super(TransBottleneckSparse, self).__init__()
#         self.conv1 = scn.NetworkInNetwork(inplanes * 4, inplanes, False)
#         self.bn1 = scn.BatchNormReLU(inplanes)
#         self.conv2 = deconv3x3_sparse(inplanes, inplanes, stride)
#         self.bn2 = scn.BatchNormReLU(inplanes)
#         self.conv3 = scn.NetworkInNetwork(inplanes, planes, False)
#         self.bn3 = scn.BatchNormalization(planes)
#         self.relu = scn.ReLU()
#         self.add = scn.AddTable()
#         self.upsample = upsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.upsample is not None:
#             residual = self.upsample(x)
#
#         out = self.add([out,residual])
#         out = self.relu(out)
#
#         return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(64, 64)
        # self.bn2 = SynchronizedBatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = conv3x3(64, 64)#128
        # self.bn3 = SynchronizedBatchNorm2d(64)#128
        # self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = self.relu1(self.bn1(self.conv1(x)))
        # x = self.relu2(self.bn2(self.conv2(x)))
        # x = self.relu3(self.bn3(self.conv3(x)))
        features.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # features.append(x)

        return features


class ResNetTranspose(nn.Module):

    def __init__(self, transblock, layers, num_classes=150):
        self.inplanes = 512
        super(ResNetTranspose, self).__init__()

        self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
        self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
        self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
        self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)

        self.skip0 = self._make_skip_layer(128, 64 * transblock.expansion)
        self.skip1 = self._make_skip_layer(256, 64 * transblock.expansion)
        self.skip2 = self._make_skip_layer(512, 128 * transblock.expansion)
        self.skip3 = self._make_skip_layer(1024, 256 * transblock.expansion)
        self.skip4 = self._make_skip_layer(2048, 512 * transblock.expansion)

        self.inplanes = 64
        self.final_conv = self._make_transpose(transblock, 64 * transblock.expansion, 3)

        self.final_deconv = nn.ConvTranspose2d(self.inplanes * transblock.expansion, num_classes, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out6_conv = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=True)
        self.out5_conv = nn.Conv2d(256 * transblock.expansion, num_classes, kernel_size=1, stride=1, bias=True)
        self.out4_conv = nn.Conv2d(128 * transblock.expansion, num_classes, kernel_size=1, stride=1, bias=True)
        self.out3_conv = nn.Conv2d(64 * transblock.expansion, num_classes, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(64 * transblock.expansion, num_classes, kernel_size=1, stride=1, bias=True)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_transpose(self, transblock, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                SynchronizedBatchNorm2d(planes),
            )
        elif self.inplanes * transblock.expansion != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes * transblock.expansion, planes,
                          kernel_size=1, stride=stride, bias=False),
                SynchronizedBatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))

        layers.append(transblock(self.inplanes, planes, stride, upsample))
        self.inplanes = planes // transblock.expansion

        return nn.Sequential(*layers)

    def _make_skip_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            SynchronizedBatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def forward(self, x, labels=None, sparse_mode=False, use_skip=True):
        [in0, in1, in2, in3, in4] = x
        if labels:
            [lab0, lab1, lab2, lab3, lab4] = labels

        out6 = self.out6_conv(in4)

        if sparse_mode:
            if labels:
                mask4 = (lab4==0).unsqueeze(1).repeat(1,in4.shape[1],1,1).type(in4.dtype)
            else:
                mask4 = (torch.argmax(out6, dim=1)==0).unsqueeze(1).repeat(1,in4.shape[1],1,1).type(in4.dtype)
            in4 = in4 * mask4

        skip4 = self.skip4(in4)
        # upsample 1
        x = self.deconv1(skip4)
        out5 = self.sigmoid(self.out5_conv(x))

        if sparse_mode:
            if labels:
                mask3 = (lab3==0).unsqueeze(1).repeat(1,in3.shape[1],1,1).type(in3.dtype)
            else:
                mask3 = (torch.argmax(out5, dim=1)==0).unsqueeze(1).repeat(1,in3.shape[1],1,1).type(in3.dtype)
            in3 = in3 * mask3

        if use_skip:
            x = x + self.skip3(in3)

        # upsample 2
        x = self.deconv2(x)
        out4 = self.sigmoid(self.out4_conv(x))

        if sparse_mode:
            if labels:
                mask2 = (lab2==0).unsqueeze(1).repeat(1,in2.shape[1],1,1).type(in2.dtype)
            else:
                mask2 = (torch.argmax(out4, dim=1)==0).unsqueeze(1).repeat(1,in2.shape[1],1,1).type(in2.dtype)
            in2 = in2 * mask2

        if use_skip:
            x = x + self.skip2(in2)

        # upsample 3
        x = self.deconv3(x)
        out3 = self.sigmoid(self.out3_conv(x))

        if sparse_mode:
            if labels:
                mask1 = (lab1==0).unsqueeze(1).repeat(1,in1.shape[1],1,1).type(in1.dtype)
            else:
                mask1 = (torch.argmax(out3, dim=1)==0).unsqueeze(1).repeat(1,in1.shape[1],1,1).type(in1.dtype)
            in1 = in1 * mask1

        if use_skip:
            x = x + self.skip1(in1)

        # upsample 4
        x = self.deconv4(x)
        out2 = self.sigmoid(self.out2_conv(x))

        if sparse_mode:
            if labels:
                mask0 = (lab0==0).unsqueeze(1).repeat(1,in0.shape[1],1,1).type(in0.dtype)
            else:
                mask0 = (torch.argmax(out2, dim=1)==0).unsqueeze(1).repeat(1,in0.shape[1],1,1).type(in0.dtype)
            in0 = in0 * mask0


        if use_skip:
            x = x + self.skip0(in0)

        # final
        x = self.final_conv(x)
        out1 = self.sigmoid(self.final_deconv(x))

        return [out6, out5, out4, out3, out2, out1]

# class ResNetSparse(nn.Module):
#
#     def __init__(self, layers):#, num_classes=1000):
#         self.inplanes = 128
#         super(ResNetSparse, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = SynchronizedBatchNorm2d(64)
#         self.relu1 = nn.ReLU(inplace=True)
#         # self.conv2 = scn.NetworkInNetwork(64, 64, True)
#         # self.bnrelu2 = scn.BatchNormReLU(64)
#         # self.conv3 = scn.NetworkInNetwork(64, 128, True)
#         # self.bnrelu2 = scn.BatchNormReLU(128)
#         self.maxpool = scn.MaxPooling(2,pool_size=3,pool_stride=2)
#
#         self.layer1 = self._make_layer(BasicBlockSparse, 64, layers[0])
#         self.layer2 = self._make_layer(BasicBlockSparse, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(BasicBlockSparse, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(BasicBlockSparse, 512, layers[3], stride=2)
#         # self.avgpool = nn.AvgPool2d(7, stride=1)
#         # self.avgpool = scn.AveragePooling(2,pool_size=7, pool_stride=1)
#         # self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.densify1 = scn.SparseToDense(2, 64)
#         self.densify2 = scn.SparseToDense(2, 128)
#         self.densify3 = scn.SparseToDense(2, 256)
#         self.densify4 = scn.SparseToDense(2, 512)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1:
#             downsample = nn.Sequential(
#                 scn.SparseToDense(2, self.inplanes),
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 scn.DenseToSparse(2),
#                 scn.BatchNormalization(planes * block.expansion),
#             )
#         elif self.inplanes != planes * block.expansion:
#             downsample = scn.Sequential(
#                 scn.NetworkInNetwork(self.inplanes, planes * block.expansion, False),
#                 scn.BatchNormalization(planes * block.expansion)
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return scn.Sequential(*layers)
#
#     def forward(self, x):
#         features = []
#         x = self.relu1(self.bn1(self.conv1(x)))
#         features.append(x)
#         x = self.dense_to_sparse(x)
#         # x = self.bnrelu2(self.conv2(x))
#         # x = self.bnrelu3(self.conv3(x))
#         # x = self.maxpool(x)
#
#         x = self.layer1(self.maxpool(x))
#         features.append(self.densify1(x))
#         x = self.layer2(x)
#         features.append(self.densify2(x))
#         x = self.layer3(x)
#         features.append(self.densify3(x))
#         x = self.layer4(x)
#         features.append(self.densify4(x))
#
#         # x = self.avgpool(x)
#         # x = self.sparse_to_dense(x)
#         # x = x.view(x.size(0), -1)
#         # x = self.fc(x)
#
#         return features

# class ResNetTransposeSparse(nn.Module):
#
#     def __init__(self, transblock, layers, num_classes=150):
#         self.inplanes = 512
#         super(ResNetTransposeSparse, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.add = AddSparseDense()
#
#         self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
#         self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
#         self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
#         self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)
#
#         self.skip0 = self._make_skip_layer(128, 64 * transblock.expansion)
#         self.skip1 = self._make_skip_layer(256, 64 * transblock.expansion)
#         self.skip2 = self._make_skip_layer(512, 128 * transblock.expansion)
#         self.skip3 = self._make_skip_layer(1024, 256 * transblock.expansion)
#         self.skip4 = self._make_skip_layer(2048, 512 * transblock.expansion)
#
#         self.densify0 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify1 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify2 = scn.SparseToDense(2, 128 * transblock.expansion)
#         self.densify3 = scn.SparseToDense(2, 256 * transblock.expansion)
#
#         self.inplanes = 64
#         self.final_conv = self._make_transpose(transblock, 64 * transblock.expansion, 3)
#
#         self.final_deconv = scn.Sequential(
#                 scn.SparseToDense(2, self.inplanes * transblock.expansion),
#                 nn.ConvTranspose2d(self.inplanes * transblock.expansion, num_classes, kernel_size=2,
#                                                stride=2, padding=0, bias=True)
#             )
#
#         self.out6_conv = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=True)
#         self.out5_conv = scn.NetworkInNetwork(256 * transblock.expansion, num_classes, True)
#         self.out4_conv = scn.NetworkInNetwork(128 * transblock.expansion, num_classes, True)
#         self.out3_conv = scn.NetworkInNetwork(64 * transblock.expansion, num_classes, True)
#         self.out2_conv = scn.NetworkInNetwork(64 * transblock.expansion, num_classes, True)
#
#         self.sparse_to_dense = scn.SparseToDense(2, num_classes)
#
#     def _make_transpose(self, transblock, planes, blocks, stride=1):
#
#         upsample = None
#         if stride != 1:
#             upsample = scn.Sequential(
#                 scn.SparseToDense(2,self.inplanes * transblock.expansion),
#                 nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
#                                   kernel_size=2, stride=stride, padding=0, bias=False),
#                 scn.DenseToSparse(2),
#                 scn.BatchNormalization(planes)
#             )
#         elif self.inplanes * transblock.expansion != planes:
#             upsample = scn.Sequential(
#                 scn.NetworkInNetwork(self.inplanes * transblock.expansion, planes, False),
#                 scn.BatchNormalization(planes)
#             )
#
#         layers = []
#
#         for i in range(1, blocks):
#             layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))
#
#         layers.append(transblock(self.inplanes, planes, stride, upsample))
#         self.inplanes = planes // transblock.expansion
#
#         return scn.Sequential(*layers)
#
#     def _make_skip_layer(self, inplanes, planes):
#
#         layers = scn.Sequential(
#             scn.NetworkInNetwork(inplanes, planes, False),
#             scn.BatchNormReLU(planes)
#         )
#         return layers
#
#     def forward(self, x, labels=None, sparse_mode=True, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#         if labels:
#             [lab0, lab1, lab2, lab3, lab4] = labels
#
#         out6 = self.out6_conv(in4)
#
#         if sparse_mode:
#             if labels:
#                 mask4 = (lab4==0).unsqueeze(1).repeat(1,in4.shape[1],1,1).type(in4.dtype)
#             else:
#                 # Rule to deactivate active site which makes the thing sparse
#                 mask4 = (torch.argmax(out6, dim=1)==0).unsqueeze(1).repeat(1,in4.shape[1],1,1).type(in4.dtype)
#             in4 = in4 * mask4
#
#         in4 = self.dense_to_sparse(in4)
#         skip4 = self.skip4(in4)
#         # upsample 1
#
#         x = self.deconv1(skip4)
#         out5 = self.sparse_to_dense(self.out5_conv(x))
#
#         if sparse_mode:
#             if labels:
#                 mask3 = (lab3==0).unsqueeze(1).repeat(1,in3.shape[1],1,1).type(in3.dtype)
#             else:
#                 mask3 = (torch.argmax(out5, dim=1)==0).unsqueeze(1).repeat(1,in3.shape[1],1,1).type(in3.dtype)
#             in3 = in3 * mask3
#
#         in3 = self.dense_to_sparse(in3)
#
#         if use_skip:
#             x = self.add([self.skip3(in3),self.densify3(x)])
#
#         # upsample 2
#         x = self.deconv2(x)
#         out4 = self.sparse_to_dense(self.out4_conv(x))
#
#         if sparse_mode:
#             if labels:
#                 mask2 = (lab2==0).unsqueeze(1).repeat(1,in2.shape[1],1,1).type(in2.dtype)
#             else:
#                 mask2 = (torch.argmax(out4, dim=1)==0).unsqueeze(1).repeat(1,in2.shape[1],1,1).type(in2.dtype)
#             in2 = in2 * mask2
#
#         in2 = self.dense_to_sparse(in2)
#
#         if use_skip:
#             x = self.add([self.skip2(in2),self.densify2(x)])
#
#         # upsample 3
#         x = self.deconv3(x)
#         out3 = self.sparse_to_dense(self.out3_conv(x))
#
#         if sparse_mode:
#             if labels:
#                 mask1 = (lab1==0).unsqueeze(1).repeat(1,in1.shape[1],1,1).type(in1.dtype)
#             else:
#                 mask1 = (torch.argmax(out3, dim=1)==0).unsqueeze(1).repeat(1,in1.shape[1],1,1).type(in1.dtype)
#             in1 = in1 * mask1
#
#         in1 = self.dense_to_sparse(in1)
#
#         if use_skip:
#             x = self.add([self.skip1(in1),self.densify1(x)])
#
#         # upsample 4
#         x = self.deconv4(x)
#         out2 = self.sparse_to_dense(self.out2_conv(x))
#
#         if sparse_mode:
#             if labels:
#                 mask0 = (lab0==0).unsqueeze(1).repeat(1,in0.shape[1],1,1).type(in0.dtype)
#             else:
#                 mask0 = (torch.argmax(out2, dim=1)==0).unsqueeze(1).repeat(1,in0.shape[1],1,1).type(in0.dtype)
#             in0 = in0 * mask0
#
#         in0 = self.dense_to_sparse(in0)
#
#         if use_skip:
#             x = self.add([self.skip0(in0),self.densify0(x)])
#
#         # final
#         x = self.final_conv(x)
#         out1 = self.final_deconv(x)
#
#         return [out6, out5, out4, out3, out2, out1]
#
# class ResNet18TransposeSparse(nn.Module):
#
#     def __init__(self, transblock, layers, num_classes=1):
#         self.inplanes = 512
#         super(ResNet18TransposeSparse, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode='nearest')
#
#         # self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#         # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
#
#         self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
#         self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
#         self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
#         self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)
#
#         # self.skip0 = self._make_skip_layer(128, 64 * transblock.expansion)
#         # self.skip1 = self._make_skip_layer(256, 64 * transblock.expansion)
#         # self.skip2 = self._make_skip_layer(512, 128 * transblock.expansion)
#         # self.skip3 = self._make_skip_layer(1024, 256 * transblock.expansion)
#         # self.skip4 = self._make_skip_layer(2048, 512 * transblock.expansion)
#
#         self.densify0 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify1 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify2 = scn.SparseToDense(2, 128 * transblock.expansion)
#         self.densify3 = scn.SparseToDense(2, 256 * transblock.expansion)
#
#         self.inplanes = 64
#         self.final_deconv = self._make_transpose(transblock, 32 * transblock.expansion, 3, stride=2)
#
#         # self.final_deconv = scn.Sequential(
#         #         # scn.SparseToDense(2, 32 * transblock.expansion),
#         #         # scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
#         #         scn.SparseToDense(2, self.inplanes * transblock.expansion),
#         #         nn.ConvTranspose2d(self.inplanes * transblock.expansion, 1, kernel_size=2,
#         #                                        stride=2, padding=0, bias=True)
#         #     )
#
#         self.out6_conv = nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=True)
#         self.out5_conv = scn.NetworkInNetwork(256 * transblock.expansion, 1, True)
#         self.out4_conv = scn.NetworkInNetwork(128 * transblock.expansion, 1, True)
#         self.out3_conv = scn.NetworkInNetwork(64 * transblock.expansion, 1, True)
#         self.out2_conv = scn.NetworkInNetwork(64 * transblock.expansion, 1, True)
#         self.out1_conv = scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
#
#         self.sparse_to_dense = scn.SparseToDense(2, num_classes)
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_transpose(self, transblock, planes, blocks, stride=1):
#
#         upsample = None
#         if stride != 1:
#             upsample = scn.Sequential(
#                 scn.SparseToDense(2,self.inplanes * transblock.expansion),
#                 nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
#                                   kernel_size=2, stride=stride, padding=0, bias=False),
#                 scn.DenseToSparse(2),
#                 scn.BatchNormalization(planes)
#             )
#         elif self.inplanes * transblock.expansion != planes:
#             upsample = scn.Sequential(
#                 scn.NetworkInNetwork(self.inplanes * transblock.expansion, planes, False),
#                 scn.BatchNormalization(planes)
#             )
#
#         layers = []
#
#         for i in range(1, blocks):
#             layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))
#
#         layers.append(transblock(self.inplanes, planes, stride, upsample))
#         self.inplanes = planes // transblock.expansion
#
#         return scn.Sequential(*layers)
#
#     def _make_skip_layer(self, inplanes, planes):
#
#         layers = scn.Sequential(
#             scn.NetworkInNetwork(inplanes, planes, False),
#             scn.BatchNormReLU(planes)
#         )
#         return layers
#
#     def _masking(self, out, crit=0.5):
#         out = 1/80 + (1/0.1 - 1/80) * out
#         a = out[:,:,0::2,0::2]
#         b = out[:,:,0::2,1::2]
#         c = out[:,:,1::2,0::2]
#         d = out[:,:,1::2,1::2]
#
#         m_max = torch.max(torch.max(torch.max(a,b),c),d)
#         m_min = torch.min(torch.min(torch.min(a,b),c),d)
#
#         mask = self.up(m_max - m_min) > crit
#
#         return mask.type(out.dtype)
#
#
#     def forward(self, x, labels=None, crit=1.0, sparse_mode=True, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [mask4, mask3, mask2, mask1, mask0] = labels
#
#         out6 = self.sigmoid(self.out6_conv(in4))
#
#         if labels is None:
#             mask4 = self._masking(out6, crit)
#             # mask4 = torch.ones_like(out6) # on force la non segmentation
#             if torch.all(mask4 == torch.zeros_like(mask4)):
#                 mask4 = (torch.rand_like(mask4) > 0.4).type(mask4.dtype)
#         in4 = in4 * mask4
#
#         in4 = self.dense_to_sparse(in4)
#         # skip4 = self.skip4(in4)
#         # upsample 1
#
#         x = self.deconv1(in4)
#         out5 = self.sigmoid(self.sparse_to_dense(self.out5_conv(x)))
#
#
#         if labels is None:
#             mask3 = self.up(mask4) * self._masking(out5, crit)
#             # mask3 = torch.ones_like(out5)
#             if torch.all(mask3 == torch.zeros_like(mask3)):
#                 mask3 = self.up(mask4 * (torch.rand_like(mask4) > 0.4).type(mask4.dtype))
#         in3 = in3 * mask3
#
#         in3 = self.dense_to_sparse(in3)
#
#         if use_skip:
#             # x = self.add([in3,x])
#             x = self.add([in3,self.densify3(x)])
#             # x = self.add([self.skip3(in3),self.densify3(x)])
#
#         # upsample 2
#         x = self.deconv2(x)
#         out4 = self.sigmoid(self.sparse_to_dense(self.out4_conv(x)))
#
#         if labels is None:
#             mask2 = self.up(mask3) * self._masking(out4, crit)
#             if torch.all(mask2 == torch.zeros_like(mask2)):
#                 mask2 = self.up(mask3 * (torch.rand_like(mask3) > 0.4).type(mask3.dtype))
#         in2 = in2 * mask2
#
#         in2 = self.dense_to_sparse(in2)
#
#         if use_skip:
#             # x = self.add([self.skip2(in2),self.densify2(x)])
#             x = self.add([in2,self.densify2(x)])
#
#         # upsample 3
#         x = self.deconv3(x)
#         out3 = self.sigmoid(self.sparse_to_dense(self.out3_conv(x)))
#
#         if labels is None:
#             mask1 = self.up(mask2) * self._masking(out3, crit)
#             if torch.all(mask1 == torch.zeros_like(mask1)):
#                 mask1 = self.up(mask2 * (torch.rand_like(mask2) > 0.4).type(mask2.dtype))
#         in1 = in1 * mask1
#
#         in1 = self.dense_to_sparse(in1)
#
#         if use_skip:
#             x = self.add([in1,self.densify1(x)])
#
#         # upsample 4
#         x = self.deconv4(x)
#         out2 = self.sigmoid(self.sparse_to_dense(self.out2_conv(x)))
#
#         if labels is None:
#             mask0 = self.up(mask1) * self._masking(out2, crit)
#             if torch.all(mask0 == torch.zeros_like(mask0)):
#                 mask0 = self.up(mask1 * (torch.rand_like(mask1) > 0.4).type(mask1.dtype))
#         in0 = in0 * mask0
#
#         in0 = self.dense_to_sparse(in0)
#
#         if use_skip:
#             # x = self.add([in0, x])
#             x = self.add([in0, self.densify0(x)])
#
#         # final
#         # x = self.final_conv(x)
#         # out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
#         # out1 = self.sigmoid(self.final_deconv(x))
#         x = self.final_deconv(x)
#         out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
#         # out1 = self.sigmoid(self.final_deconv(x))
#
#
#         return [out6, out5, out4, out3, out2, out1], [mask4, mask3, mask2, mask1, mask0]
#
# class QuadtreeDepthDecoder(nn.Module):
#
#     def __init__(self, ch=[512,256,128,64,64,64]):
#         # mobilenet_inplanes = [160, 112, 40, 24, 16, 16]
#         self.inplanes = 512
#         super(QuadtreeDepthDecoder, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.sparse_to_dense = scn.SparseToDense(2, 2)
#
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode="nearest")
#
#         self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
#         self.out4_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[1], nOut=2, filter_size=1, bias=False)
#         self.out3_conv = scn.SubmanifoldConvolution(2, ch[2], 2, 1, False)
#         self.out2_conv = scn.SubmanifoldConvolution(2, ch[3], 2, 1, False)
#         self.out1_conv = scn.SubmanifoldConvolution(2, ch[4], 2, 1, False)
#         self.out0_conv = scn.SubmanifoldConvolution(2, ch[5], 1, 1, False)
#
#         i=0
#         self.deconv4_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn4_0 = scn.BatchNormalization(ch[i+1], leakiness=0) # leakiness=0 implit bn + ReLU
#         self.densify4 = scn.SparseToDense(2, ch[i+1])
#         self.deconv4_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn4_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=1
#         self.deconv3_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn3_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify3 = scn.SparseToDense(2, ch[i+1])
#         self.deconv3_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn3_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=2
#         self.deconv2_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn2_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify2 = scn.SparseToDense(2, ch[i+1])
#         self.deconv2_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn2_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=3
#         self.deconv1_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn1_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify1 = scn.SparseToDense(2, ch[i+1])
#         self.deconv1_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn1_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=4
#         self.deconv0_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn0_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify0 = scn.SparseToDense(2, ch[i+1])
#         self.deconv0_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn0_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         self.densify_out = scn.SparseToDense(2,1)
#
#
#         self.sigmoid = nn.Sigmoid()
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#
#
#
#     def _make_layer(self, inplanes, outplanes, use_skip):
#         conv1 = conv3x3_sparse(in_planes, out_planes)
#         bn1 = scn.BatchNormalization(outplanes, leakiness=0)
#
#
#
#     def forward(self, x, labels=None, crit=1.0, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [label5, label4, label3, label2, label1] = labels
#
#         # tic = time.time()
#         out5 = self.out5_conv(in4) # 6x20
#         # print(f"out5: {time.time() - tic}s")
#         # tic = time.time()
#         disp5, mask5 = torch.unbind(out5, 1)
#         disp5 = self.sigmoid(disp5.unsqueeze(1))
#         mask5 = self.sigmoid(mask5.unsqueeze(1))
#         # print(f"unbind: {time.time() - tic}s")
#         # tic = time.time()
#
#         if labels is None:
#             # non differentiable: impossible lors de l'entrainement
#             in4 = in4 * torch.round(mask5)
#         else:
#             in4 = in4 * label5
#
#         # print(f"labels: {time.time() - tic}s")
#         # tic = time.time()
#
#
#         # x = self.upsample(in4)
#         # x = self.dense_to_sparse(x)
#         #
#         # x = self.deconv4_0(x)
#         # x = self.bn4_0(x)
#         # x = self.deconv4_1(x)
#         # x = self.bn4_1(x)
#
#         in4 = self.dense_to_sparse(in4)
#         # print(f"dense_to_sparse: {time.time() - tic}s")
#         # tic = time.time()
#
#         x = self.deconv4_0(in4)
#         # print(f"deconv0: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn4_0(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.densify4(x)
#         # print(f"densify: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.upsample(x)
#         # print(f"upsample: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.dense_to_sparse(x)
#         # print(f"dense to sparse: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.deconv4_1(x)
#         # print(f"deconv1: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn4_1(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#
#         out4 = self.sparse_to_dense(self.out4_conv(x)) # 12x40
#         # print(f"out4: {time.time() - tic}s")
#         # tic = time.time()
#         disp4, mask4 = torch.unbind(out4, 1)
#         disp4 = self.sigmoid(disp4.unsqueeze(1))
#         mask4 = self.sigmoid(mask4.unsqueeze(1))
#
#         if labels is None:
#             in3 = in3 * torch.round(mask4)
#         else:
#             in3 = in3 * label4
#         # print(f"unbind + labels: {time.time() - tic}s")
#         # tic = time.time()
#
#         in3 = self.dense_to_sparse(in3)
#
#         # print(f"dense_to_sparse: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.add([in3, self.densify4(x)])
#         # print(f"add: {time.time() - tic}s")
#         # tic = time.time()
#         #
#         # x = self.densify3(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         # x = self.deconv3_0(x)
#         # x = self.bn3_0(x)
#         # x = self.deconv3_1(x)
#         # x = self.bn3_1(x)
#
#         x = self.deconv3_0(x)
#         # print(f"deconv3-0: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn3_0(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.densify3(x)
#         # print(f"densify: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.upsample(x)
#         # print(f"upsample: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.dense_to_sparse(x)
#         # print(f"dense_to_sparse: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.deconv3_1(x)
#         # print(f"deconv3-1: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn3_1(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#
#         out3 = self.sparse_to_dense(self.out3_conv(x)) # 24x80
#         # print(f"out3: {time.time() - tic}s")
#         # tic = time.time()
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = self.sigmoid(disp3.unsqueeze(1))
#         mask3 = self.sigmoid(mask3.unsqueeze(1))
#
#         if labels is None:
#             in2 = in2 * torch.round(mask3)
#         else:
#             in2 = in2 * label3
#         # print(f"unbind + labels: {time.time() - tic}s")
#         # tic = time.time()
#
#         in2 = self.dense_to_sparse(in2)
#         # print(f"dense to sparse: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.add([in2, self.densify3(x)])
#         # print(f"add: {time.time() - tic}s")
#         # tic = time.time()
#
#
#         # x = self.densify2(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         # x = self.deconv2_0(x)
#         # x = self.bn2_0(x)
#         # x = self.deconv2_1(x)
#         # x = self.bn2_1(x)
#
#         x = self.deconv2_0(x)
#         # print(f"deconv2-0: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn2_0(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.densify2(x)
#         # print(f"densify: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.upsample(x)
#         # print(f"upsample: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.dense_to_sparse(x)
#         # print(f"dense_to_sparse: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.deconv2_1(x)
#         # print(f"deconv2-1: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn2_1(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#
#         out2 = self.sparse_to_dense(self.out2_conv(x)) # 24x80
#         # print(f"sparse to dense: {time.time() - tic}s")
#         # tic = time.time()
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = self.sigmoid(disp2.unsqueeze(1))
#         mask2 = self.sigmoid(mask2.unsqueeze(1))
#
#         if labels is None:
#             in1 = in1 * torch.round(mask2)
#         else:
#             in1 = in1 * label2
#
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#
#         in1 = self.dense_to_sparse(in1)
#         # print(f"dense to sparse: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.add([in1, self.densify2(x)])
#         # print(f"add: {time.time() - tic}s")
#         # tic = time.time()
#
#         # x = self.densify1(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         # x = self.deconv1_0(x)
#         # x = self.bn1_0(x)
#         # x = self.deconv1_1(x)
#         # x = self.bn1_1(x)
#
#         x = self.deconv1_0(x)
#         # print(f"deconv1-0: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn1_0(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.densify1(x)
#         # print(f"densify: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.upsample(x)
#         # print(f"upsample: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.dense_to_sparse(x)
#         # print(f"dense to sparse: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.deconv1_1(x)
#         # print(f"deconv1-1: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn1_1(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#
#         out1 = self.sparse_to_dense(self.out1_conv(x)) # 24x80
#         # print(f"sparse to dense: {time.time() - tic}s")
#         # tic = time.time()
#         disp1, mask1 = torch.unbind(out1, 1)
#
#         # print(f"unbind: {time.time() - tic}s")
#         # tic = time.time()
#         disp1 = self.sigmoid(disp1.unsqueeze(1))
#         mask1 = self.sigmoid(mask1.unsqueeze(1))
#
#         # print(f"out1: {time.time() - tic}s")
#         # tic = time.time()
#
#         if labels is None:
#             in0 = in0 * torch.round(mask1)
#         else:
#             in0 = in0 * label1
#
#         # print(f"labels: {time.time() - tic}s")
#         # tic = time.time()
#
#         in0 = self.dense_to_sparse(in0)
#
#         # print(f"dense_to_sparse: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.add([in0, self.densify0(x)])
#
#         # print(f"add: {time.time() - tic}s")
#         # tic = time.time()
#
#
#         # x = self.densify0(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         # x = self.deconv0_0(x)
#         # x = self.bn0_0(x)
#         # x = self.deconv0_1(x)
#         # x = self.bn0_1(x)
#
#         x = self.deconv0_0(x)
#
#         # print(f"deconv0-0: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn0_0(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.densify0(x)
#         # print(f"densify0: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.upsample(x)
#         # print(f"upsample: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.dense_to_sparse(x)
#         # print(f"dense to sparse: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.deconv0_1(x)
#         # print(f"deconv0-1: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.bn0_1(x)
#         # print(f"batchNorm: {time.time() - tic}s")
#         # tic = time.time()
#
#         disp0 = self.densify_out(self.out0_conv(x)) # 24x80
#         # print(f"disp0: {time.time() - tic}s")
#         # tic = time.time()
#
#
#         # x = self.final_deconv(x)
#         # out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
#
#         # for disp in [disp5, disp4, disp3, disp2, disp1, disp0]:
#         #     print(disp.size())
#
#
#         return [disp5, disp4, disp3, disp2, disp1, disp0], [mask5, mask4, mask3, mask2, mask1]


# class QuadtreeDepthDecoderLight(nn.Module):
#
#     def __init__(self, ch=[512,256,128,64,64,64]):
#         # mobilenet_inplanes = [160, 112, 40, 24, 16, 16]
#         self.inplanes = 512
#         super(QuadtreeDepthDecoderLight, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.sparse_to_dense = scn.SparseToDense(2, 2)
#
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode="nearest")
#
#         # self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
#         # self.out4_conv = nn.Conv2d(ch[1], 2, kernel_size=1, stride=1, bias=True)
#         # self.out4_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[1], nOut=2, filter_size=1, bias=True)
#         # self.out3_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[2], nOut=2, filter_size=1, bias=True)
#         self.out3_conv = nn.Conv2d(ch[2], 2, kernel_size=1, stride=1, bias=True)
#         self.out2_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[3], nOut=2, filter_size=1, bias=True)
#         self.out1_conv = scn.SubmanifoldConvolution(2, ch[4], 2, 1, True)
#         self.out0_conv = scn.SubmanifoldConvolution(2, ch[5], 1, 1, True)
#
#         i=0
#         self.deconv4_0 = conv3x3(ch[i], ch[i+1])
#         self.bn4_0 = SynchronizedBatchNorm2d(ch[i+1]) # leakiness=0 implit bn + ReLU
#         # self.densify4 = scn.SparseToDense(2, ch[i+1])
#         self.deconv4_1 = conv3x3(ch[i+1], ch[i+1])
#         self.bn4_1 = SynchronizedBatchNorm2d(ch[i+1])
#
#         i=1
#         self.deconv3_0 = conv3x3(ch[i], ch[i+1])
#         self.bn3_0 = SynchronizedBatchNorm2d(ch[i+1])
#         # self.densify3 = scn.SparseToDense(2, ch[i+1])
#         self.deconv3_1 = conv3x3(ch[i+1], ch[i+1])
#         self.bn3_1 = SynchronizedBatchNorm2d(ch[i+1])
#
#         i=2
#         self.deconv2_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn2_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify2 = scn.SparseToDense(2, ch[i+1])
#         self.deconv2_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn2_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=3
#         self.deconv1_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn1_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify1 = scn.SparseToDense(2, ch[i+1])
#         self.deconv1_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn1_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=4
#         self.deconv0_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn0_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify0 = scn.SparseToDense(2, ch[i+1])
#         self.deconv0_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn0_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         self.densify_out = scn.SparseToDense(2,1)
#
#
#         self.sigmoid = nn.Sigmoid()
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#
#
#
#     def _make_layer(self, inplanes, outplanes, use_skip):
#         conv1 = conv3x3_sparse(in_planes, out_planes)
#         bn1 = scn.BatchNormalization(outplanes, leakiness=0)
#
#
#
#     def forward(self, x, labels=None, crit=1.0, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [label3, label2, label1] = labels
#
#         # first layer is dense
#         x = self.deconv4_0(in4)
#         x = self.bn4_0(x)
#         x = self.upsample(x)
#         x = self.deconv4_1(x)
#         x = self.bn4_1(x)
#
#
#         x = x + in3
#
#         # second layer is dense
#         x = self.deconv3_0(x)
#         x = self.bn3_0(x)
#         x = self.upsample(x)
#         x = self.deconv3_1(x)
#         x = self.bn3_1(x)
#
#         out3 = self.out3_conv(x)
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = self.sigmoid(disp3.unsqueeze(1))
#         mask3 = self.sigmoid(mask3.unsqueeze(1))
#
#         if labels is None:
#             in2 = in2 * torch.round(mask3)
#         else:
#             in2 = in2 * label3
#
#         in2 = self.dense_to_sparse(in2)
#         x = self.add([in2, x])
#
#         x = self.deconv2_0(x)
#         x = self.bn2_0(x)
#         x = self.densify2(x)
#         x = self.upsample(x)
#         x = self.dense_to_sparse(x)
#         x = self.deconv2_1(x)
#         x = self.bn2_1(x)
#
#         out2 = self.sparse_to_dense(self.out2_conv(x))
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = self.sigmoid(disp2.unsqueeze(1))
#         mask2 = self.sigmoid(mask2.unsqueeze(1))
#
#         if labels is None:
#             in1 = in1 * torch.round(mask2)
#         else:
#             in1 = in1 * label2
#
#
#         in1 = self.dense_to_sparse(in1)
#         x = self.add([in1, self.densify2(x)])
#
#         x = self.deconv1_0(x)
#         x = self.bn1_0(x)
#         x = self.densify1(x)
#         x = self.upsample(x)
#         x = self.dense_to_sparse(x)
#         x = self.deconv1_1(x)
#         x = self.bn1_1(x)
#
#         out1 = self.sparse_to_dense(self.out1_conv(x))
#         disp1, mask1 = torch.unbind(out1, 1)
#
#         disp1 = self.sigmoid(disp1.unsqueeze(1))
#         mask1 = self.sigmoid(mask1.unsqueeze(1))
#
#
#         if labels is None:
#             in0 = in0 * torch.round(mask1)
#         else:
#             in0 = in0 * label1
#
#
#         in0 = self.dense_to_sparse(in0)
#
#         x = self.add([in0, self.densify0(x)])
#
#
#         x = self.deconv0_0(x)
#         x = self.bn0_0(x)
#         x = self.densify0(x)
#         x = self.upsample(x)
#         x = self.dense_to_sparse(x)
#         x = self.deconv0_1(x)
#         x = self.bn0_1(x)
#
#         disp0 = self.densify_out(self.out0_conv(x))
#
#
#         return [disp3, disp2, disp1, disp0], [mask3, mask2, mask1]

class QuadtreeDepthDecoderSpConv(nn.Module):

    def __init__(self, ch=[512,256,128,64,64,64]):
        super(QuadtreeDepthDecoderSpConv, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.toDense = spconv.ToDense()
        self.sigmoid = nn.Sigmoid()

        self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
        self.out4_conv = spconv.SubMConv2d(ch[1], 2, 1, 1)
        self.out3_conv = spconv.SubMConv2d(ch[2], 2, 1, 1)
        self.out2_conv = spconv.SubMConv2d(ch[3], 2, 1, 1)
        self.out1_conv = spconv.SubMConv2d(ch[4], 2, 1, 1)
        self.out0_conv = spconv.SubMConv2d(ch[5], 1, 1, 1)

        self.layers = {}
        self.out_conv = {}

        i = 0
        self.layers4_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers4_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 1
        self.layers3_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers3_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 2
        self.layers2_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers2_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 3
        self.layers1_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers1_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 4
        self.layers0_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers0_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))


    def toSparse(self, x, mask=None):
        if mask is None:
            x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,1))
        else:
            # tic = time.time()
            b,ch,h,w = x.size()
            ms = torch.nonzero(mask.squeeze(1))
            x = x.permute(0,2,3,1)
            x = x[ms[:,0],ms[:,1], ms[:,2],:]
            ms = ms.contiguous().int()
            x = spconv.SparseConvTensor(x.reshape(-1,ch), ms, (h,w), b)
            # print(f"toSparse += {time.time() - tic}")
        return x

    def sparsify(self, x, out):
        ## inputs:
        # x: SparseConvTensor -> sparse features
        # out: SparseConvTensor -> sparse outputs f
        # tic = time.time()
        idx = out.indices
        y = self.sigmoid(out.features[:,1]) > 0.5
        feat = x.features[y]
        ind = idx[y].contiguous().int()
        x = spconv.SparseConvTensor(feat, ind, x.spatial_shape, x.batch_size)
        # print(f"sparsify += {time.time() - tic}")
        return x

    def add(self, x, dense_features):
        # tic = time.time()
        idx = x.indices.type(torch.LongTensor).to(dense_features.device)
        dense_features = dense_features.permute(0,2,3,1)
        dense_features = dense_features[idx[:,0],idx[:,1], idx[:,2],:].reshape(-1, dense_features.size()[3])

        x = spconv.SparseConvTensor(x.features + dense_features, x.indices, x.spatial_shape, x.batch_size)

        # print(f"add += {time.time() - tic}")
        return x


    def forward(self, features, labels=None, crit=1.0, use_skip=True):
        [in0, in1, in2, in3, in4] = features


        if labels is not None:
            [label5, label4, label3, label2, label1] = labels

        disp = {}
        mask = {}

        out = self.sigmoid(self.out5_conv(in4))
        d, m = torch.unbind(out, 1)
        disp[5] = d.unsqueeze(1)
        mask[5] = m.unsqueeze(1)

        ## Layer 4
        x = in4
        if labels is None:
            label5 = torch.round(mask[5])
        x = x * label5

        if torch.all(label5 == 0):
            b,_,h,w = label1.size()
            for i in range(5):
                disp[i] = 0.5 * torch.ones(b,1,h*2**(i+1),w*2**(i+1))
                if i > 0:
                    mask[i] = torch.zeros(b,1,h*2**(i+1),w*2**(i+1))
            return [disp[i] for i in range(5,-1,-1)], [mask[i] for i in range(5,0,-1)]
        else:
            x = self.toSparse(x, label5)
            x = self.layers4_0(x)
            x = self.toDense(x)
            x = self.upsample(x)
            x = self.toSparse(x, self.upsample(label5))
            x = self.layers4_1(x)


            out = self.out4_conv(x)
            d, m = torch.unbind(self.sigmoid(self.toDense(out)), 1)
            disp[4] = d.unsqueeze(1)
            mask[4] = m.unsqueeze(1)


        ## Layer 3
        if labels is None:
            label4 = torch.round(mask[4])
            x = self.sparsify(x, out)
            x = self.add(x, in3)
        else:
            x = self.toDense(x) + in3
            x = x * label4
            x = self.toSparse(x)


        if torch.all(label4 == 0):
            b,_,h,w = label1.size()
            for i in range(4):
                disp[i] = 0.5 * torch.ones(b,1,h*2**(i+1),w*2**(i+1))
                if i > 0:
                    mask[i] = torch.zeros(b,1,h*2**(i+1),w*2**(i+1))
            return [disp[i] for i in range(5,-1,-1)], [mask[i] for i in range(5,0,-1)]
        else:
            x = self.layers3_0(x)
            x = self.toDense(x)
            x = self.upsample(x)
            x = self.toSparse(x, self.upsample(label4))
            x = self.layers3_1(x)

            out = self.out3_conv(x)
            d, m = torch.unbind(self.sigmoid(self.toDense(out)), 1)
            disp[3] = d.unsqueeze(1)
            mask[3] = m.unsqueeze(1)


        ## Layer 2
        if labels is None:
            label3 = torch.round(mask[3])
            x = self.sparsify(x, out)
            x = self.add(x, in2)
        else:
            x = self.toDense(x) + in2
            x = x * label3
            x = self.toSparse(x)


        if torch.all(label3 == 0):
            b,_,h,w = label1.size()
            for i in range(3):
                disp[i] = 0.5 * torch.ones(b,1,h*2**(i+1),w*2**(i+1))
                if i > 0:
                    mask[i] = torch.zeros(b,1,h*2**(i+1),w*2**(i+1))
            return [disp[i] for i in range(5,-1,-1)], [mask[i] for i in range(5,0,-1)]
        else:
            x = self.layers2_0(x)
            x = self.toDense(x)
            x = self.upsample(x)
            x = self.toSparse(x, self.upsample(label3))
            x = self.layers2_1(x)

            out = self.out2_conv(x)
            d, m = torch.unbind(self.sigmoid(self.toDense(out)), 1)
            disp[2] = d.unsqueeze(1)
            mask[2] = m.unsqueeze(1)


        ## Layer 1

        if labels is None:
            label2 = torch.round(mask[2])
            x = self.sparsify(x, out)
            x = self.add(x, in1)
        else:
            x = self.toDense(x) + in1
            x = x * label2
            x = self.toSparse(x, label2)

        if torch.all(label2 == 0):
            b,_,h,w = label1.size()
            for i in range(2):
                disp[i] = 0.5 * torch.ones(b,1,h*2**(i+1),w*2**(i+1))
                if i > 0:
                    mask[i] = torch.zeros(b,1,h*2**(i+1),w*2**(i+1))
            return [disp[i] for i in range(5,-1,-1)], [mask[i] for i in range(5,0,-1)]
        else:
            x = self.layers1_0(x)
            x = self.toDense(x)
            x = self.upsample(x)
            x = self.toSparse(x, self.upsample(label2))
            x = self.layers1_1(x)


            out = self.out1_conv(x)
            d, m = torch.unbind(self.sigmoid(self.toDense(out)), 1)
            disp[1] = d.unsqueeze(1)
            mask[1] = m.unsqueeze(1)


        ## Layer 0
        if labels is None:
            label1 = torch.round(mask[1])
            x = self.sparsify(x, out)
            x = self.add(x, in0)
        else:
            x = self.toDense(x) + in0
            x = x * label1
            x = self.toSparse(x, label1)

        if torch.all(label1 == 0):
            b,_,h,w = label1.size()
            disp[0] = 0.5 * torch.ones(b,1,h*2**2,w*2**2)
            return [disp[i] for i in range(5,-1,-1)], [mask[i] for i in range(5,0,-1)]
        else:
            x = self.layers0_0(x)
            x = self.toDense(x)
            x = self.upsample(x)
            x = self.toSparse(x, self.upsample(label1))
            x = self.layers0_1(x)

            disp[0] = self.sigmoid(self.toDense(self.out0_conv(x)))


        return [disp[i] for i in range(5,-1,-1)], [mask[i] for i in range(5,0,-1)]

class QuadtreeDepthDecoderSpConv_old(nn.Module):

    def __init__(self, ch=[512,256,128,64,64,64]):
        super(QuadtreeDepthDecoderSpConv_old, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.toDense = spconv.ToDense()
        self.sigmoid = nn.Sigmoid()

        self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
        self.out4_conv = spconv.SubMConv2d(ch[1], 2, 1, 1)
        self.out3_conv = spconv.SubMConv2d(ch[2], 2, 1, 1)
        self.out2_conv = spconv.SubMConv2d(ch[3], 2, 1, 1)
        self.out1_conv = spconv.SubMConv2d(ch[4], 2, 1, 1)
        self.out0_conv = spconv.SubMConv2d(ch[5], 1, 1, 1)

        self.layers = {}
        self.out_conv = {}

        i = 0
        self.layers4_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers4_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 1
        self.layers3_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers3_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 2
        self.layers2_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers2_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 3
        self.layers1_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers1_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 4
        self.layers0_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.layers0_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        # self.layers_0 = {}
        # self.layers_1 = {}
        #
        # for i in range(4,-1,-1):
        #     self.layers_0[i] = spconv.SparseSequential(
        #                             spconv.SubMConv2d(ch[4-i], ch[5-i], 3, 1),
        #                             SynchronizedBatchNorm2d(ch[5-i]))
        #     self.layers_1[i] = spconv.SparseSequential(
        #                             spconv.SubMConv2d(ch[5-i], ch[5-i], 3, 1),
        #                             SynchronizedBatchNorm2d(ch[5-i]))
        #     if i == 0:
        #         self.out_conv[i] = spconv.SubMConv2d(ch[5-i], 1, 1, 1)
        #     else:
        #         self.out_conv[i] = spconv.SubMConv2d(ch[5-i], 2, 1, 1)

    def toSparse(self, x):
        return spconv.SparseConvTensor.from_dense(x.permute(0,2,3,1))

    def forward(self, features, labels=None, crit=1.0, use_skip=True):
        [in0, in1, in2, in3, in4] = features


        if labels is not None:
            [label5, label4, label3, label2, label1] = labels

        disp = {}
        mask = {}

        # out = self.out5_conv(features[4])
        out = self.out5_conv(in4)
        d, m = torch.unbind(out, 1)
        disp[5] = self.sigmoid(d.unsqueeze(1))
        mask[5] = self.sigmoid(m.unsqueeze(1))

        ## Layer 4
        x = in4
        if labels is None:
            x = x * torch.round(mask[5])
        else:
            x = x * label5

        x = self.toSparse(x)
        x = self.layers4_0(x)
        x = self.toDense(x)
        x = self.upsample(x)
        x = self.toSparse(x)
        x = self.layers4_1(x)

        out = self.toDense(self.out4_conv(x))
        d, m = torch.unbind(out, 1)
        disp[4] = self.sigmoid(d.unsqueeze(1))
        mask[4] = self.sigmoid(m.unsqueeze(1))


        ## Layer 3
        x = self.toDense(x) + in3
        if labels is None:
            x = x * torch.round(mask[4])
        else:
            x = x * label4

        x = self.toSparse(x)
        x = self.layers3_0(x)
        x = self.toDense(x)
        x = self.upsample(x)
        x = self.toSparse(x)
        x = self.layers3_1(x)

        out = self.toDense(self.out3_conv(x))
        d, m = torch.unbind(out, 1)
        disp[3] = self.sigmoid(d.unsqueeze(1))
        mask[3] = self.sigmoid(m.unsqueeze(1))


        ## Layer 2
        x = self.toDense(x) + in2
        if labels is None:
            x = x * torch.round(mask[3])
        else:
            x = x * label3

        x = self.toSparse(x)
        x = self.layers2_0(x)
        x = self.toDense(x)
        x = self.upsample(x)
        x = self.toSparse(x)
        x = self.layers2_1(x)

        out = self.toDense(self.out2_conv(x))
        d, m = torch.unbind(out, 1)
        disp[2] = self.sigmoid(d.unsqueeze(1))
        mask[2] = self.sigmoid(m.unsqueeze(1))


        ## Layer 1
        x = self.toDense(x) + in1
        if labels is None:
            x = x * torch.round(mask[2])
        else:
            x = x * label2

        x = self.toSparse(x)
        x = self.layers1_0(x)
        x = self.toDense(x)
        x = self.upsample(x)
        x = self.toSparse(x)
        x = self.layers1_1(x)

        out = self.toDense(self.out1_conv(x))
        d, m = torch.unbind(out, 1)
        disp[1] = self.sigmoid(d.unsqueeze(1))
        mask[1] = self.sigmoid(m.unsqueeze(1))


        ## Layer 0
        x = self.toDense(x) + in0
        if labels is None:
            x = x * torch.round(mask[1])
        else:
            x = x * label1

        x = self.toSparse(x)
        x = self.layers0_0(x)
        x = self.toDense(x)
        x = self.upsample(x)
        x = self.toSparse(x)
        x = self.layers0_1(x)

        disp[0] = self.sigmoid(self.toDense(self.out0_conv(x)))
        # d, m = torch.unbind(out, 1)
        # disp[0] = self.sigmoid(d.unsqueeze(1))
        # mask[0] = self.sigmoid(m.unsqueeze(1))


        # for i in range(4,-1,-1):
        #     if i == 4:
        #         x = features[i]
        #     else:
        #         x = x + features[i]
        #
        #     if labels is None:
        #         x = x * torch.round(mask[i+1])
        #     else:
        #         x = x * labels[4-i]
        #
        #     if torch.all(x == 0):
        #         b,_,h,w = x.size()
        #         disp[i] = torch.zeros(b, 1, h*2, w*2)
        #         mask[i] = torch.zeros(b, 1, h*2, w*2)
        #         print("Hello there! All zeros.")
        #     else:
        #         x = self.toSparse(x)
        #         x = self.layers[(i,0)](x)
        #         x = self.toDense(x)
        #         x = self.upsample(x)
        #         x = self.toSparse(x)
        #         x = self.layers[(i,1)](x)
        #
        #         out = self.toDense(self.out_conv[i](x))
        #         if i == 0:
        #             disp[i] = self.sigmoid(out.unsqueeze(1))
        #         else:
        #             d, m = torch.unbind(out, 1)
        #             disp[i] = self.sigmoid(d.unsqueeze(1))
        #             mask[i] = self.sigmoid(m.unsqueeze(1))
        #             x = self.toDense(x)

        return [disp[i] for i in range(5,-1,-1)], [mask[i] for i in range(5,0,-1)]



class QuadtreeDepthDecoderSpConv2(nn.Module):

    def __init__(self, ch=[512,256,128,64,64,64], isTrain=False):
        super(QuadtreeDepthDecoderSpConv2, self).__init__()

        self.isTrain = isTrain
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.toDense = spconv.ToDense()
        self.sigmoid = nn.Sigmoid()

        self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
        self.out4_conv = spconv.SubMConv2d(ch[1], 2, 1, 1)
        self.out3_conv = spconv.SubMConv2d(ch[2], 2, 1, 1)
        self.out2_conv = spconv.SubMConv2d(ch[3], 2, 1, 1)
        self.out1_conv = spconv.SubMConv2d(ch[4], 2, 1, 1)
        self.out0_conv = spconv.SubMConv2d(ch[5], 1, 1, 1)

        self.layers = {}
        self.out_conv = {}

        i = 0
        self.layers4_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.upconv4 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers4_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 1
        self.layers3_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.upconv3 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers3_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 2
        self.layers2_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.upconv2 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers2_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 3
        self.layers1_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.upconv1 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers1_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 4
        self.layers0_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        self.upconv0 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers0_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))


    def toSparse(self, x, mask=None):
        if mask is None:
            x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,1))
            return x
        elif self.isTrain:
            x = x * mask
            x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,1))
            return x
        else: # faster but not differentiable solution
            b,ch,h,w = x.size()
            ms = torch.nonzero(mask.squeeze(1))
            x = x.permute(0,2,3,1)
            x = x[ms[:,0],ms[:,1], ms[:,2],:]
            ms = ms.contiguous().int()
            x = spconv.SparseConvTensor(x.reshape(-1,ch), ms, (h,w), b)
        return x

    # def toSparse(self, x, mask=None):
    #     x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,1))
    #     return x

    def add(self, x, dense_features):
        idx = x.indices.type(torch.LongTensor).to(dense_features.device)
        dense_features = dense_features.permute(0,2,3,1)
        dense_features = dense_features[idx[:,0],idx[:,1], idx[:,2],:].reshape(-1, dense_features.size()[3])

        # x = x.replace_feature(x.features + dense_features
        x = spconv.SparseConvTensor(x.features + dense_features, x.indices, x.spatial_shape, x.batch_size)
        return x


    def forward(self, features, labels=None, crit=1.0, use_skip=True):
        [in0, in1, in2, in3, in4] = features


        if labels is not None:
            [label5, label4, label3, label2, label1] = labels

        disp = {}
        mask = {}

        out = self.out5_conv(in4)
        d, m = torch.unbind(out, 1)
        disp[5] = self.sigmoid(d.unsqueeze(1))
        mask[5] = self.sigmoid(m.unsqueeze(1))

        ## Layer 4
        x = in4
        if labels is None:
            label5 = torch.round(mask[5])
        # x = x * label5

        x = self.toSparse(x, label5)
        x = self.layers4_0(x)
        x = self.upconv4(x)
        x = self.add(x, in3)
        x = self.layers4_1(x)

        out = self.toDense(self.out4_conv(x))
        d, m = torch.unbind(out, 1)
        disp[4] = self.sigmoid(d.unsqueeze(1))
        mask[4] = self.sigmoid(m.unsqueeze(1))


        ## Layer 3
        x = self.toDense(x)
        if labels is None:
            label4 = torch.round(mask[4])

        x = self.toSparse(x, label4)
        x = self.layers3_0(x)
        x = self.upconv3(x)
        x = self.add(x, in2)
        x = self.layers3_1(x)

        out = self.toDense(self.out3_conv(x))
        d, m = torch.unbind(out, 1)
        disp[3] = self.sigmoid(d.unsqueeze(1))
        mask[3] = self.sigmoid(m.unsqueeze(1))


        ## Layer 2
        x = self.toDense(x)
        if labels is None:
            label3 = torch.round(mask[3])

        x = self.toSparse(x, label3)
        x = self.layers2_0(x)
        x = self.upconv2(x)
        x = self.add(x, in1)
        x = self.layers2_1(x)

        out = self.toDense(self.out2_conv(x))
        d, m = torch.unbind(out, 1)
        disp[2] = self.sigmoid(d.unsqueeze(1))
        mask[2] = self.sigmoid(m.unsqueeze(1))


        ## Layer 1
        x = self.toDense(x)
        if labels is None:
            label2 = torch.round(mask[2])

        x = self.toSparse(x, label2)
        x = self.layers1_0(x)
        x = self.upconv1(x)
        x = self.add(x, in0)
        x = self.layers1_1(x)

        out = self.toDense(self.out1_conv(x))
        d, m = torch.unbind(out, 1)
        disp[1] = self.sigmoid(d.unsqueeze(1))
        mask[1] = self.sigmoid(m.unsqueeze(1))


        ## Layer 0
        x = self.toDense(x)
        if labels is None:
            label1 = torch.round(mask[1])

        x = self.toSparse(x, label1)
        x = self.layers0_0(x)
        x = self.upconv0(x)
        x = self.layers0_1(x)

        disp[0] = self.sigmoid(self.toDense(self.out0_conv(x)))
        # if torch.all(label1 == 0):
        #     b,_,h,w = x.size()
        #     disp[0] = 0.5 * torch.ones(b,1,h*2,w*2).to(x.device)
        # else:
        #     x = self.toSparse(x, label1)
        #     x = self.layers0_0(x)
        #     x = self.upconv0(x)
        #     x = self.layers0_1(x)
        #
        #     disp[0] = self.sigmoid(self.toDense(self.out0_conv(x)))

        return [disp[i] for i in range(5,-1,-1)], [mask[i] for i in range(5,0,-1)]


# replace upconv by upsample
class QuadtreeDepthDecoderSpConv3(nn.Module):

    def __init__(self, ch=[512,256,128,64,64,64]):
        super(QuadtreeDepthDecoderSpConv3, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.toDense = spconv.ToDense()
        self.sigmoid = nn.Sigmoid()

        self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
        self.out4_conv = spconv.SubMConv2d(ch[1], 2, 1, 1)
        self.out3_conv = spconv.SubMConv2d(ch[2], 2, 1, 1)
        self.out2_conv = spconv.SubMConv2d(ch[3], 2, 1, 1)
        self.out1_conv = spconv.SubMConv2d(ch[4], 2, 1, 1)
        self.out0_conv = spconv.SubMConv2d(ch[5], 1, 1, 1)

        self.layers = {}
        self.out_conv = {}

        i = 0
        self.layers4_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        # self.upconv4 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers4_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 1
        self.layers3_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        # self.upconv3 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers3_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 2
        self.layers2_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        # self.upconv2 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers2_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 3
        self.layers1_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        # self.upconv1 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers1_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))

        i = 4
        self.layers0_0 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))
        # self.upconv0 = spconv.SparseConvTranspose2d(ch[i+1], ch[i+1], 2, 2)
        self.layers0_1 = spconv.SparseSequential(
                            spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                            SynchronizedBatchNorm2d(ch[i+1]))


    def toSparse(self, x, mask=None):
        if mask is None:
            x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,1))
            return x
        else:
            x = x * mask
            x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,1))
            return x

    # def toSparse(self, x, mask=None):
    #     x = spconv.SparseConvTensor.from_dense(x.permute(0,2,3,1))
    #     return x

    # def add(self, x, dense_features):
    #     idx = x.indices.type(torch.LongTensor).to(dense_features.device)
    #     dense_features = dense_features.permute(0,2,3,1)
    #     dense_features = dense_features[idx[:,0],idx[:,1], idx[:,2],:].reshape(-1, dense_features.size()[3])
    #
    #     # x = x.replace_feature(x.features + dense_features
    #     x = spconv.SparseConvTensor(x.features + dense_features, x.indices, x.spatial_shape, x.batch_size)
    #     return x

    def add(self, x, dense_features):
        x = self.toDense(x)
        x = self.upsample(x)
        x = x + dense_features
        x = self.toSparse(x, x > 0)
        return x


    def forward(self, features, labels=None, crit=1.0, use_skip=True):
        [in0, in1, in2, in3, in4] = features


        if labels is not None:
            [label5, label4, label3, label2, label1] = labels

        disp = {}
        mask = {}

        out = self.out5_conv(in4)
        d, m = torch.unbind(out, 1)
        disp[5] = self.sigmoid(d.unsqueeze(1))
        mask[5] = self.sigmoid(m.unsqueeze(1))

        ## Layer 4
        x = in4
        if labels is None:
            label5 = torch.round(mask[5])
        # x = x * label5

        x = self.toSparse(x, label5)
        x = self.layers4_0(x)
        x = self.add(x, in3)
        x = self.layers4_1(x)

        out = self.toDense(self.out4_conv(x))
        d, m = torch.unbind(out, 1)
        disp[4] = self.sigmoid(d.unsqueeze(1))
        mask[4] = self.sigmoid(m.unsqueeze(1))


        ## Layer 3
        x = self.toDense(x)
        if labels is None:
            label4 = torch.round(mask[4])

        x = self.toSparse(x, label4)
        x = self.layers3_0(x)
        x = self.add(x, in2)
        x = self.layers3_1(x)

        out = self.toDense(self.out3_conv(x))
        d, m = torch.unbind(out, 1)
        disp[3] = self.sigmoid(d.unsqueeze(1))
        mask[3] = self.sigmoid(m.unsqueeze(1))


        ## Layer 2
        x = self.toDense(x)
        if labels is None:
            label3 = torch.round(mask[3])

        x = self.toSparse(x, label3)
        x = self.layers2_0(x)
        x = self.add(x, in1)
        x = self.layers2_1(x)

        out = self.toDense(self.out2_conv(x))
        d, m = torch.unbind(out, 1)
        disp[2] = self.sigmoid(d.unsqueeze(1))
        mask[2] = self.sigmoid(m.unsqueeze(1))


        ## Layer 1
        x = self.toDense(x)
        if labels is None:
            label2 = torch.round(mask[2])

        x = self.toSparse(x, label2)
        x = self.layers1_0(x)
        x = self.add(x, in0)
        x = self.layers1_1(x)

        out = self.toDense(self.out1_conv(x))
        d, m = torch.unbind(out, 1)
        disp[1] = self.sigmoid(d.unsqueeze(1))
        mask[1] = self.sigmoid(m.unsqueeze(1))


        ## Layer 0
        x = self.toDense(x)
        if labels is None:
            label1 = torch.round(mask[1])

        x = self.toSparse(x, label1)
        x = self.layers0_0(x)
        x = self.toDense(x)
        x = self.upsample(x)
        x = self.toSparse(x)
        x = self.layers0_1(x)

        disp[0] = self.sigmoid(self.toDense(self.out0_conv(x)))

        return [disp[i] for i in range(5,-1,-1)], [mask[i] for i in range(5,0,-1)]



class QuadtreeDepthDecoderLightSpConv(nn.Module):

    def __init__(self, ch=[512,256,128,64,64,64]):
        # mobilenet_inplanes = [160, 112, 40, 24, 16, 16]
        self.inplanes = 512
        super(QuadtreeDepthDecoderLightSpConv, self).__init__()

        # self.dense_to_sparse = scn.DenseToSparse(2)
        # self.sparse_to_dense = scn.SparseToDense(2, 2)

        # self.add = AddSparseDense()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.out3_conv = nn.Conv2d(ch[2], 2, kernel_size=1, stride=1, bias=True)
        self.out2_conv = spconv.SubMConv2d(ch[3], 2, 1, 1)
        self.out1_conv = spconv.SubMConv2d(ch[4], 2, 1, 1)
        self.out0_conv = spconv.SubMConv2d(ch[5], 1, 1, 1)

        i=0
        self.deconv4_0 = conv3x3(ch[i], ch[i+1])
        self.bn4_0 = SynchronizedBatchNorm2d(ch[i+1])
        self.deconv4_1 = conv3x3(ch[i+1], ch[i+1])
        self.bn4_1 = SynchronizedBatchNorm2d(ch[i+1])

        i=1
        self.deconv3_0 = conv3x3(ch[i], ch[i+1])
        self.bn3_0 = SynchronizedBatchNorm2d(ch[i+1])
        self.deconv3_1 = conv3x3(ch[i+1], ch[i+1])
        self.bn3_1 = SynchronizedBatchNorm2d(ch[i+1])

        i=2
        self.layer2_0 = spconv.SparseSequential(
                                spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                                SynchronizedBatchNorm2d(ch[i+1]))
        self.layer2_1 = spconv.SparseSequential(
                                spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                                SynchronizedBatchNorm2d(ch[i+1])
        )

        i=3
        self.layer1_0 = spconv.SparseSequential(
                                spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                                SynchronizedBatchNorm2d(ch[i+1]))
        self.layer1_1 = spconv.SparseSequential(
                                spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                                # spconv.SparseInverseConv2d(ch[i+1], ch[i+1], 3, 2),
                                SynchronizedBatchNorm2d(ch[i+1])
        )


        i=4
        self.layer0_0 = spconv.SparseSequential(
                                spconv.SubMConv2d(ch[i], ch[i+1], 3, 1),
                                SynchronizedBatchNorm2d(ch[i+1]))
        self.layer0_1 = spconv.SparseSequential(
                                spconv.SubMConv2d(ch[i+1], ch[i+1], 3, 1),
                                # spconv.SparseInverseConv2d(ch[i+1], ch[i+1], 3, 2),
                                SynchronizedBatchNorm2d(ch[i+1])
        )


        self.densify_out = spconv.ToDense()
        self.toDense = spconv.ToDense()


        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')



    def _make_layer(self, inplanes, outplanes, use_skip):
        conv1 = conv3x3_sparse(in_planes, out_planes)
        bn1 = scn.BatchNormalization(outplanes, leakiness=0)

    def toSparse(self, x):
        return spconv.SparseConvTensor.from_dense(x.permute(0,2,3,1))


    def forward(self, x, labels=None, crit=1.0, use_skip=True):
        [in0, in1, in2, in3, in4] = x

        if labels is not None:
            [label3, label2, label1] = labels

        # first layer is dense
        x = self.deconv4_0(in4)
        x = self.bn4_0(x)
        x = self.upsample(x)
        x = self.deconv4_1(x)
        x = self.bn4_1(x)


        x = x + in3

        # second layer is dense
        x = self.deconv3_0(x)
        x = self.bn3_0(x)
        x = self.upsample(x)
        x = self.deconv3_1(x)
        x = self.bn3_1(x)

        out3 = self.out3_conv(x)
        disp3, mask3 = torch.unbind(out3, 1)
        disp3 = self.sigmoid(disp3.unsqueeze(1))
        mask3 = self.sigmoid(mask3.unsqueeze(1))

        x = x + in2
        if labels is None:
            x = x * torch.round(mask3)
        else:
            x = x * label3

        x = self.toSparse(x)

        x = self.layer2_0(x)
        x = self.toDense(x)
        x = self.upsample(x)
        x = self.toSparse(x)
        x = self.layer2_1(x)

        out2 = self.toDense(self.out2_conv(x))
        disp2, mask2 = torch.unbind(out2, 1)
        disp2 = self.sigmoid(disp2.unsqueeze(1))
        mask2 = self.sigmoid(mask2.unsqueeze(1))


        x = in1 + self.toDense(x)

        if labels is None:
            x = x * torch.round(mask2)
        else:
            x = x * label2

        x = self.toSparse(x)


        x = self.layer1_0(x)
        x = self.toDense(x)
        x = self.upsample(x)
        x = self.toSparse(x)
        x = self.layer1_1(x)

        out1 = self.toDense(self.out1_conv(x))
        disp1, mask1 = torch.unbind(out1, 1)

        disp1 = self.sigmoid(disp1.unsqueeze(1))
        mask1 = self.sigmoid(mask1.unsqueeze(1))

        x = in0 + self.toDense(x)
        if labels is None:
            x = x * torch.round(mask1)
        else:
            x = x * label1

        x = self.toSparse(x)
        x = self.layer0_0(x)
        x = self.toDense(x)
        x = self.upsample(x)
        x = self.toSparse(x)
        x = self.layer0_1(x)

        disp0 = self.sigmoid(self.toDense(self.out0_conv(x)))


        return [disp3, disp2, disp1, disp0], [mask3, mask2, mask1]


# # Avoid having to switch between dense and sparse
# class QuadtreeDepthDecoderLight2(nn.Module):
#
#     def __init__(self, ch=[512,256,128,64,64,64]):
#         # mobilenet_inplanes = [160, 112, 40, 24, 16, 16]
#         self.inplanes = 512
#         super(QuadtreeDepthDecoderLight2, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.sparse_to_dense = scn.SparseToDense(2, 2)
#
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode="nearest")
#
#         # self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
#         # self.out4_conv = nn.Conv2d(ch[1], 2, kernel_size=1, stride=1, bias=True)
#         # self.out4_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[1], nOut=2, filter_size=1, bias=True)
#         # self.out3_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[2], nOut=2, filter_size=1, bias=True)
#         self.out3_conv = nn.Conv2d(ch[2], 2, kernel_size=1, stride=1, bias=True)
#         self.out2_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[3], nOut=2, filter_size=1, bias=True)
#         self.out1_conv = scn.SubmanifoldConvolution(2, ch[4], 2, 1, True)
#         self.out0_conv = scn.SubmanifoldConvolution(2, ch[5], 1, 1, True)
#
#         i=0
#         self.deconv4_0 = conv3x3(ch[i], ch[i+1])
#         self.bn4_0 = SynchronizedBatchNorm2d(ch[i+1]) # leakiness=0 implit bn + ReLU
#         # self.densify4 = scn.SparseToDense(2, ch[i+1])
#         self.deconv4_1 = conv3x3(ch[i+1], ch[i+1])
#         self.bn4_1 = SynchronizedBatchNorm2d(ch[i+1])
#
#         i=1
#         self.deconv3_0 = conv3x3(ch[i], ch[i+1])
#         self.bn3_0 = SynchronizedBatchNorm2d(ch[i+1])
#         # self.densify3 = scn.SparseToDense(2, ch[i+1])
#         self.deconv3_1 = conv3x3(ch[i+1], ch[i+1])
#         self.bn3_1 = SynchronizedBatchNorm2d(ch[i+1])
#
#         i=2
#         self.deconv2_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn2_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify2 = scn.SparseToDense(2, ch[i+1])
#         self.deconv2_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn2_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=3
#         self.deconv1_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn1_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify1 = scn.SparseToDense(2, ch[i+1])
#         self.deconv1_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn1_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=4
#         self.deconv0_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn0_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify0 = scn.SparseToDense(2, ch[i+1])
#         self.deconv0_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn0_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         self.densify_out = scn.SparseToDense(2,1)
#
#
#         self.sigmoid = nn.Sigmoid()
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsample_sparse = scn.UnPooling(dimension=2, pool_size=2, pool_stride=2)
#
#
#
#     def _make_layer(self, inplanes, outplanes, use_skip):
#         conv1 = conv3x3_sparse(in_planes, out_planes)
#         bn1 = scn.BatchNormalization(outplanes, leakiness=0)
#
#
#
#     def forward(self, x, labels=None, crit=1.0, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [label3, label2, label1] = labels
#
#         # first layer is dense
#         x = self.deconv4_0(in4)
#         x = self.bn4_0(x)
#         x = self.upsample(x)
#         x = self.deconv4_1(x)
#         x = self.bn4_1(x)
#
#
#         x = x + in3
#
#         # second layer is dense
#         x = self.deconv3_0(x)
#         x = self.bn3_0(x)
#         x = self.upsample(x)
#         x = self.deconv3_1(x)
#         x = self.bn3_1(x)
#
#         out3 = self.out3_conv(x)
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = self.sigmoid(disp3.unsqueeze(1))
#         mask3 = self.sigmoid(mask3.unsqueeze(1))
#
#         if labels is None:
#             in2 = in2 * torch.round(mask3)
#         else:
#             in2 = in2 * label3
#
#         in2 = self.dense_to_sparse(in2)
#         x = self.add([in2, x])
#
#
#         x = self.deconv2_0(x)
#         x = self.bn2_0(x)
#         # x = self.densify2(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         print(x.spatial_size)
#         print(x.features.shape)
#         print(x.get_spatial_locations())
#         x = self.upsample_sparse(x)
#         print(x.spatial_size)
#         print(x.features.shape)
#         print(x.get_spatial_locations())
#         x = self.deconv2_1(x)
#         x = self.bn2_1(x)
#
#         out2 = self.sparse_to_dense(self.out2_conv(x))
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = self.sigmoid(disp2.unsqueeze(1))
#         mask2 = self.sigmoid(mask2.unsqueeze(1))
#
#         print(f"disp2.size(): {disp2.size()}")
#
#         if labels is None:
#             in1 = in1 * torch.round(mask2)
#         else:
#             in1 = in1 * label2
#
#
#         in1 = self.dense_to_sparse(in1)
#         x = self.add([in1, self.densify2(x)])
#
#         x = self.deconv1_0(x)
#         x = self.bn1_0(x)
#         # x = self.densify1(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         x = self.upsample_sparse(x)
#         x = self.deconv1_1(x)
#         x = self.bn1_1(x)
#
#         out1 = self.sparse_to_dense(self.out1_conv(x))
#         disp1, mask1 = torch.unbind(out1, 1)
#
#         disp1 = self.sigmoid(disp1.unsqueeze(1))
#         mask1 = self.sigmoid(mask1.unsqueeze(1))
#
#
#         if labels is None:
#             in0 = in0 * torch.round(mask1)
#         else:
#             in0 = in0 * label1
#
#
#         in0 = self.dense_to_sparse(in0)
#
#         x = self.add([in0, self.densify0(x)])
#
#
#         x = self.deconv0_0(x)
#         x = self.bn0_0(x)
#         # x = self.densify0(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         x = self.upsample_sparse(x)
#         x = self.deconv0_1(x)
#         x = self.bn0_1(x)
#
#         disp0 = self.densify_out(self.out0_conv(x))
#
#
#         return [disp3, disp2, disp1, disp0], [mask3, mask2, mask1]



# class QuadtreeDepthDecoderV2(nn.Module):
#
#     def __init__(self, ch=[512,256,128,64,64,64]):
#         # mobilenet_inplanes = [160, 112, 40, 24, 16, 16]
#         self.inplanes = 512
#         super(QuadtreeDepthDecoderV2, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.sparse_to_dense = scn.SparseToDense(2, 2)
#
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode="nearest")
#
#         self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
#         self.out4_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[1], nOut=2, filter_size=1, bias=False)
#         self.out3_conv = scn.SubmanifoldConvolution(2, ch[2], 2, 1, False)
#         self.out2_conv = scn.SubmanifoldConvolution(2, ch[3], 2, 1, False)
#         self.out1_conv = scn.SubmanifoldConvolution(2, ch[4], 2, 1, False)
#         self.out0_conv = scn.SubmanifoldConvolution(2, ch[5], 1, 1, False)
#
#         i=0
#         self.deconv4_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn4_0 = scn.BatchNormalization(ch[i+1], leakiness=0) # leakiness=0 implit bn + ReLU
#         self.densify4 = scn.SparseToDense(2, ch[i+1])
#         self.deconv4_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn4_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=1
#         self.deconv3_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn3_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify3 = scn.SparseToDense(2, ch[i+1])
#         self.deconv3_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn3_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=2
#         self.deconv2_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn2_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify2 = scn.SparseToDense(2, ch[i+1])
#         self.deconv2_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn2_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=3
#         self.deconv1_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn1_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify1 = scn.SparseToDense(2, ch[i+1])
#         self.deconv1_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn1_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         i=4
#         self.deconv0_0 = conv3x3_sparse(ch[i], ch[i+1])
#         self.bn0_0 = scn.BatchNormalization(ch[i+1], leakiness=0)
#         self.densify0 = scn.SparseToDense(2, ch[i+1])
#         self.deconv0_1 = conv3x3_sparse(ch[i+1], ch[i+1])
#         self.bn0_1 = scn.BatchNormalization(ch[i+1], leakiness=0)
#
#         self.densify_out = scn.SparseToDense(2,1)
#
#
#         self.sigmoid = nn.Sigmoid()
#         # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.unpooling = scn.UnPooling(dimension=2, pool_size=2, pool_stride=2)
#
#
#
#     def _make_layer(self, inplanes, outplanes, use_skip):
#         conv1 = conv3x3_sparse(in_planes, out_planes)
#         bn1 = scn.BatchNormalization(outplanes, leakiness=0)
#
#
#
#     def forward(self, x, labels=None, crit=1.0, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [label5, label4, label3, label2, label1] = labels
#
#         out5 = self.out5_conv(in4) # 6x20
#         disp5, mask5 = torch.unbind(out5, 1)
#         disp5 = self.sigmoid(disp5.unsqueeze(1))
#         mask5 = self.sigmoid(mask5.unsqueeze(1))
#
#         if labels is None:
#             # non differentiable: impossible lors de l'entrainement
#             in4 = in4 * torch.round(mask5)
#         else:
#             in4 = in4 * label5
#
#
#         # x = self.upsample(in4)
#         # x = self.dense_to_sparse(x)
#         #
#         # x = self.deconv4_0(x)
#         # x = self.bn4_0(x)
#         # x = self.deconv4_1(x)
#         # x = self.bn4_1(x)
#
#         in4 = self.dense_to_sparse(in4)
#
#         x = self.deconv4_0(in4)
#         x = self.bn4_0(x)
#         # print(f"features: {x.features.shape}")
#         # x = self.densify4(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         test0 = self.densify4(x)
#         # print(test0.size())
#         x = self.unpooling(x)
#         test = self.densify4(x)
#         # print(test.size())
#         # print(f"spatial size: {x.spatial_size}")
#         # print(f"metadata: {x.metadata}")
#         # print(f"features: {x.features.shape}")
#         x = self.deconv4_1(x)
#         x = self.bn4_1(x)
#
#         out4 = self.sparse_to_dense(self.out4_conv(x)) # 12x40
#         # print(out4.size())
#         disp4, mask4 = torch.unbind(out4, 1)
#         disp4 = self.sigmoid(disp4.unsqueeze(1))
#         mask4 = self.sigmoid(mask4.unsqueeze(1))
#
#         if labels is None:
#             in3 = in3 * torch.round(mask4)
#         else:
#             in3 = in3 * label4
#
#         in3 = self.dense_to_sparse(in3)
#         x = self.add([in3, self.densify4(x)])
#         #
#         # x = self.densify3(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         # x = self.deconv3_0(x)
#         # x = self.bn3_0(x)
#         # x = self.deconv3_1(x)
#         # x = self.bn3_1(x)
#
#         x = self.deconv3_0(x)
#         x = self.bn3_0(x)
#         # x = self.densify3(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         x = self.unpooling(x)
#         x = self.deconv3_1(x)
#         x = self.bn3_1(x)
#
#         out3 = self.sparse_to_dense(self.out3_conv(x)) # 24x80
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = self.sigmoid(disp3.unsqueeze(1))
#         mask3 = self.sigmoid(mask3.unsqueeze(1))
#
#         if labels is None:
#             in2 = in2 * torch.round(mask3)
#         else:
#             in2 = in2 * label3
#
#         in2 = self.dense_to_sparse(in2)
#         x = self.add([in2, self.densify3(x)])
#
#
#         x = self.deconv2_0(x)
#         x = self.bn2_0(x)
#         # x = self.densify2(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         x = self.unpooling(x)
#         x = self.deconv2_1(x)
#         x = self.bn2_1(x)
#
#         out2 = self.sparse_to_dense(self.out2_conv(x)) # 24x80
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = self.sigmoid(disp2.unsqueeze(1))
#         mask2 = self.sigmoid(mask2.unsqueeze(1))
#
#         if labels is None:
#             in1 = in1 * torch.round(mask2)
#         else:
#             in1 = in1 * label2
#
#         in1 = self.dense_to_sparse(in1)
#         x = self.add([in1, self.densify2(x)])
#
#         # x = self.densify1(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         # x = self.deconv1_0(x)
#         # x = self.bn1_0(x)
#         # x = self.deconv1_1(x)
#         # x = self.bn1_1(x)
#
#         x = self.deconv1_0(x)
#         x = self.bn1_0(x)
#         # x = self.densify1(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         x = self.unpooling(x)
#         x = self.deconv1_1(x)
#         x = self.bn1_1(x)
#
#         out1 = self.sparse_to_dense(self.out1_conv(x)) # 24x80
#         disp1, mask1 = torch.unbind(out1, 1)
#         disp1 = self.sigmoid(disp1.unsqueeze(1))
#         mask1 = self.sigmoid(mask1.unsqueeze(1))
#
#         if labels is None:
#             in0 = in0 * torch.round(mask1)
#         else:
#             in0 = in0 * label1
#
#         in0 = self.dense_to_sparse(in0)
#         x = self.add([in0, self.densify0(x)])
#
#
#         # x = self.densify0(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         # x = self.deconv0_0(x)
#         # x = self.bn0_0(x)
#         # x = self.deconv0_1(x)
#         # x = self.bn0_1(x)
#
#         x = self.deconv0_0(x)
#         x = self.bn0_0(x)
#         # x = self.densify0(x)
#         # x = self.upsample(x)
#         # x = self.dense_to_sparse(x)
#         x = self.unpooling(x)
#         x = self.deconv0_1(x)
#         x = self.bn0_1(x)
#
#         disp0 = self.densify_out(self.out0_conv(x)) # 24x80
#
#
#         # x = self.final_deconv(x)
#         # out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
#
#         # for disp in [disp5, disp4, disp3, disp2, disp1, disp0]:
#         #     print(disp.size())
#
#
#         return [disp5, disp4, disp3, disp2, disp1, disp0], [mask5, mask4, mask3, mask2, mask1]



# class QGNDepthDecoder_new(nn.Module):
#
#     def __init__(self, ch=[512,256,128,64,64,64]):
#         # mobilenet_inplanes = [160, 112, 40, 24, 16, 16]
#         self.inplanes = 512
#         super(QGNDepthDecoder_new, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.sparse_to_dense = scn.SparseToDense(2, 2)
#
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode="nearest")
#
#         self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
#         self.out4_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[1], nOut=2, filter_size=1, bias=False)
#         self.out3_conv = scn.SubmanifoldConvolution(2, ch[2], 2, 1, False)
#         self.out2_conv = scn.SubmanifoldConvolution(2, ch[3], 2, 1, False)
#         self.out1_conv = scn.SubmanifoldConvolution(2, ch[4], 2, 1, False)
#         self.out0_conv = scn.SubmanifoldConvolution(2, ch[5], 1, 1, False)
#
#         i=0
#         self.deconv4_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv4_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify4 = scn.SparseToDense(2, ch[i+1])
#
#         i=1
#         self.deconv3_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv3_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify3 = scn.SparseToDense(2, ch[i+1])
#
#         i=2
#         self.deconv2_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv2_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify2 = scn.SparseToDense(2, ch[i+1])
#
#         i=3
#         self.deconv1_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv1_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify1 = scn.SparseToDense(2, ch[i+1])
#
#         i=4
#         self.deconv0_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv0_1 = self._Conv(ch[i+1], ch[i+1])
#
#         self.densify_out = scn.SparseToDense(2,1)
#
#
#         self.sigmoid = nn.Sigmoid()
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#
#
#
#     def _make_layer(self, inplanes, outplanes):
#         conv1 = conv3x3_sparse(in_planes, out_planes)
#         bn1 = scn.BatchNormalization(outplanes, leakiness=0)
#
#     def _UpConv_old(self, inplanes, outplanes):
#         return scn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
#                         scn.DenseToSparse(2),
#                         conv3x3_sparse(inplanes, outplanes),
#                         scn.BatchNormalization(outplanes, leakiness=0))
#
#     def _UpConv(self, inplanes, outplanes):
#         return scn.Sequential(conv3x3_sparse(inplanes, outplanes),
#                         scn.BatchNormalization(outplanes, leakiness=0),
#                         scn.UnPooling(2, 2, 2))
#
#     def _Conv(self, inplanes, outplanes):
#         return scn.Sequential(conv3x3_sparse(inplanes, outplanes),
#                         scn.BatchNormalization(outplanes, leakiness=0))
#
#
#
#
#     def forward(self, x, labels=None, crit=1.0, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [label5, label4, label3, label2, label1] = labels
#
#         #dense
#         tic = time.time()
#         out5 = self.sigmoid(self.out5_conv(in4)) # 6x20
#         print(f"Q5: {time.time() - tic}s")
#         tic = time.time()
#         disp5, mask5 = torch.unbind(out5, 1)
#         disp5 = disp5.unsqueeze(1)
#         mask5 = mask5.unsqueeze(1)
#         print(f"unbind: {time.time() - tic}s")
#         tic = time.time()
#
#         if labels is None:
#             # non differentiable: impossible lors de l'entrainement
#             in4 = in4 * torch.round(mask5)
#         else:
#             in4 = in4 * label5
#         print(f"labels: {time.time() - tic}s")
#         tic = time.time()
#
#         x = self.deconv4_0(self.dense_to_sparse(in4))
#         print(f"deconv0: {time.time() - tic}s")
#         tic = time.time()
#         x = self.add([x, in3])
#         print(f"add: {time.time() - tic}s")
#         tic = time.time()
#         x = self.deconv4_1(x)
#         print(f"deconv1: {time.time() - tic}s")
#         tic = time.time()
#
#         print(f"1st layer: {time.time() - tic}s")
#         tic = time.time()
#
#
#         out4 = self.sigmoid(self.sparse_to_dense(self.out4_conv(x))) # 12x40
#         print(f"Q4: {time.time() - tic}s")
#         tic = time.time()
#         disp4, mask4 = torch.unbind(out4, 1)
#         disp4 = disp4.unsqueeze(1)
#         mask4 = mask4.unsqueeze(1)
#
#         x = self.densify4(x)
#         if labels is None:
#             x = x * torch.round(mask4)
#         else:
#             x = x * label4
#
#         x = self.deconv3_0(x)
#         x = self.add([x, in2])
#         x = self.deconv3_1(x)
#
#         print(f"2nd layer: {time.time() - tic}s")
#         tic = time.time()
#
#         out3 = self.sigmoid(self.sparse_to_dense(self.out3_conv(x))) # 24x80
#         print(f"Q3: {time.time() - tic}s")
#         tic = time.time()
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = disp3.unsqueeze(1)
#         mask3 = mask3.unsqueeze(1)
#
#         x = self.densify3(x)
#         if labels is None:
#             x = x * torch.round(mask3)
#         else:
#             x = x * label3
#
#         x = self.deconv2_0(x)
#         x = self.add([x, in1])
#         x = self.deconv2_1(x)
#
#         print(f"3rd layer: {time.time() - tic}s")
#         tic = time.time()
#
#         out2 = self.sigmoid(self.sparse_to_dense(self.out2_conv(x))) # 48x160
#         print(f"Q2: {time.time() - tic}s")
#         tic = time.time()
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = disp2.unsqueeze(1)
#         mask2 = mask2.unsqueeze(1)
#
#         x = self.densify2(x)
#         if labels is None:
#             x = x * torch.round(mask2)
#         else:
#             x = x * label2
#
#         x = self.deconv1_0(x)
#         x = self.add([x, in0])
#         x = self.deconv1_1(x)
#
#         print(f"4th layer: {time.time() - tic}s")
#         tic = time.time()
#
#         out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x))) # 96x320
#         print(f"Q1: {time.time() - tic}s")
#         tic = time.time()
#         disp1, mask1 = torch.unbind(out1, 1)
#         disp1 = disp1.unsqueeze(1)
#         mask1 = mask1.unsqueeze(1)
#
#         x = self.densify1(x)
#         if labels is None:
#             x = x * torch.round(mask1)
#         else:
#             x = x * label1
#
#         x = self.deconv0_0(x)
#         x = self.deconv0_1(x)
#         print(f"5th layer: {time.time() - tic}s")
#         tic = time.time()
#
#         disp0 = self.sigmoid(self.densify_out(self.out0_conv(x))) # 192x640
#         print(f"Q0: {time.time() - tic}s")
#
#
#
#         return [disp5, disp4, disp3, disp2, disp1, disp0], [mask5, mask4, mask3, mask2, mask1]

# class QGNDepthDecoder(nn.Module):
#
#     def __init__(self, ch=[512,256,128,64,64,64]):
#         # mobilenet_inplanes = [160, 112, 40, 24, 16, 16]
#         self.inplanes = 512
#         super(QGNDepthDecoder, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.sparse_to_dense = scn.SparseToDense(2, 2)
#
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode="nearest")
#
#         self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
#         self.out4_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[1], nOut=2, filter_size=1, bias=False)
#         self.out3_conv = scn.SubmanifoldConvolution(2, ch[2], 2, 1, False)
#         self.out2_conv = scn.SubmanifoldConvolution(2, ch[3], 2, 1, False)
#         self.out1_conv = scn.SubmanifoldConvolution(2, ch[4], 2, 1, False)
#         self.out0_conv = scn.SubmanifoldConvolution(2, ch[5], 1, 1, False)
#
#         i=0
#         self.deconv4_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv4_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify4 = scn.SparseToDense(2, ch[i+1])
#
#         i=1
#         self.deconv3_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv3_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify3 = scn.SparseToDense(2, ch[i+1])
#
#         i=2
#         self.deconv2_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv2_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify2 = scn.SparseToDense(2, ch[i+1])
#
#         i=3
#         self.deconv1_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv1_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify1 = scn.SparseToDense(2, ch[i+1])
#
#         i=4
#         self.deconv0_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv0_1 = self._Conv(ch[i+1], ch[i+1])
#
#         self.densify_out = scn.SparseToDense(2,1)
#
#
#         self.sigmoid = nn.Sigmoid()
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#
#
#
#     def _make_layer(self, inplanes, outplanes):
#         conv1 = conv3x3_sparse(in_planes, out_planes)
#         bn1 = scn.BatchNormalization(outplanes, leakiness=0)
#
#     def _UpConv(self, inplanes, outplanes):
#         return scn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
#                         scn.DenseToSparse(2),
#                         conv3x3_sparse(inplanes, outplanes),
#                         scn.BatchNormalization(outplanes, leakiness=0))
#
#
#     def _Conv(self, inplanes, outplanes):
#         return scn.Sequential(conv3x3_sparse(inplanes, outplanes),
#                         scn.BatchNormalization(outplanes, leakiness=0))
#
#
#
#
#     def forward(self, x, labels=None, crit=1.0, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [label5, label4, label3, label2, label1] = labels
#
#         #dense
#         # tic = time.time()
#         out5 = self.sigmoid(self.out5_conv(in4)) # 6x20
#         # print(f"Q5: {time.time() - tic}s")
#         # tic = time.time()
#         disp5, mask5 = torch.unbind(out5, 1)
#         disp5 = disp5.unsqueeze(1)
#         mask5 = mask5.unsqueeze(1)
#         # print(f"unbind: {time.time() - tic}s")
#         # tic = time.time()
#
#         if labels is None:
#             # non differentiable: impossible lors de l'entrainement
#             in4 = in4 * torch.round(mask5)
#         else:
#             in4 = in4 * label5
#         # print(f"labels: {time.time() - tic}s")
#         # tic = time.time()
#
#         x = self.deconv4_0(in4)
#         # print(f"deconv0: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.add([x, in3])
#         # print(f"add: {time.time() - tic}s")
#         # tic = time.time()
#         x = self.deconv4_1(x)
#         # print(f"deconv1: {time.time() - tic}s")
#         # tic = time.time()
#
#         # print(f"1st layer: {time.time() - tic}s")
#         # tic = time.time()
#
#
#         out4 = self.sigmoid(self.sparse_to_dense(self.out4_conv(x))) # 12x40
#         # print(f"Q4: {time.time() - tic}s")
#         # tic = time.time()
#         disp4, mask4 = torch.unbind(out4, 1)
#         disp4 = disp4.unsqueeze(1)
#         mask4 = mask4.unsqueeze(1)
#
#         # tic = time.time()
#         x = self.densify4(x)
#         # print(f"densify: {time.time() - tic}s")
#         # tic = time.time()
#         if labels is None:
#             x = x * torch.round(mask4)
#         else:
#             x = x * label4
#
#         x = self.deconv3_0(x)
#         x = self.add([x, in2])
#         x = self.deconv3_1(x)
#
#         # print(f"2nd layer: {time.time() - tic}s")
#         # tic = time.time()
#
#         out3 = self.sigmoid(self.sparse_to_dense(self.out3_conv(x))) # 24x80
#         # print(f"Q3: {time.time() - tic}s")
#         # tic = time.time()
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = disp3.unsqueeze(1)
#         mask3 = mask3.unsqueeze(1)
#
#         x = self.densify3(x)
#         if labels is None:
#             x = x * torch.round(mask3)
#         else:
#             x = x * label3
#
#         x = self.deconv2_0(x)
#         x = self.add([x, in1])
#         x = self.deconv2_1(x)
#
#         # print(f"3rd layer: {time.time() - tic}s")
#         # tic = time.time()
#
#         out2 = self.sigmoid(self.sparse_to_dense(self.out2_conv(x))) # 48x160
#         # print(f"Q2: {time.time() - tic}s")
#         # tic = time.time()
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = disp2.unsqueeze(1)
#         mask2 = mask2.unsqueeze(1)
#
#         x = self.densify2(x)
#         if labels is None:
#             x = x * torch.round(mask2)
#         else:
#             x = x * label2
#
#         x = self.deconv1_0(x)
#         x = self.add([x, in0])
#         x = self.deconv1_1(x)
#
#         # print(f"4th layer: {time.time() - tic}s")
#         # tic = time.time()
#
#         out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x))) # 96x320
#         # print(f"Q1: {time.time() - tic}s")
#         # tic = time.time()
#         disp1, mask1 = torch.unbind(out1, 1)
#         disp1 = disp1.unsqueeze(1)
#         mask1 = mask1.unsqueeze(1)
#
#         x = self.densify1(x)
#         if labels is None:
#             x = x * torch.round(mask1)
#         else:
#             x = x * label1
#
#         x = self.deconv0_0(x)
#         x = self.deconv0_1(x)
#         # print(f"5th layer: {time.time() - tic}s")
#         # tic = time.time()
#
#         disp0 = self.sigmoid(self.densify_out(self.out0_conv(x))) # 192x640
#         # print(f"Q0: {time.time() - tic}s")
#
#
#
#         return [disp5, disp4, disp3, disp2, disp1, disp0], [mask5, mask4, mask3, mask2, mask1]
#
#
# class QGNDepthDecoder4(nn.Module):
#
#     def __init__(self, ch=[512,256,128,64,64,64]):
#         # mobilenet_inplanes = [160, 112, 40, 24, 16, 16]
#         self.inplanes = 512
#         super(QGNDepthDecoder4, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.sparse_to_dense = scn.SparseToDense(2, 2)
#
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode="nearest")
#
#         # self.out5_conv = nn.Conv2d(ch[0], 2, kernel_size=1, stride=1, bias=True)
#         self.out4_conv = nn.Conv2d(ch[1], 2, kernel_size=1, stride=1, bias=True)
#         # self.out4_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[1], nOut=2, filter_size=1, bias=False)
#         self.out3_conv = scn.SubmanifoldConvolution(dimension=2, nIn=ch[2], nOut=2, filter_size=1, bias=False)
#         self.out2_conv = scn.SubmanifoldConvolution(2, ch[3], 2, 1, False)
#         self.out1_conv = scn.SubmanifoldConvolution(2, ch[4], 2, 1, False)
#         self.out0_conv = scn.SubmanifoldConvolution(2, ch[5], 1, 1, False)
#
#         i=0
#         self.deconv4_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv4_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify4 = scn.SparseToDense(2, ch[i+1])
#
#         i=1
#         self.deconv3_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv3_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify3 = scn.SparseToDense(2, ch[i+1])
#
#         i=2
#         self.deconv2_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv2_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify2 = scn.SparseToDense(2, ch[i+1])
#
#         i=3
#         self.deconv1_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv1_1 = self._Conv(ch[i+1], ch[i+1])
#         self.densify1 = scn.SparseToDense(2, ch[i+1])
#
#         i=4
#         self.deconv0_0 = self._UpConv(ch[i], ch[i+1])
#         self.deconv0_1 = self._Conv(ch[i+1], ch[i+1])
#
#         self.densify_out = scn.SparseToDense(2,1)
#
#
#         self.sigmoid = nn.Sigmoid()
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#
#
#
#     def _make_layer(self, inplanes, outplanes):
#         conv1 = conv3x3_sparse(in_planes, out_planes)
#         bn1 = scn.BatchNormalization(outplanes, leakiness=0)
#
#     def _UpConv(self, inplanes, outplanes):
#         return scn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
#                         scn.DenseToSparse(2),
#                         conv3x3_sparse(inplanes, outplanes),
#                         scn.BatchNormalization(outplanes, leakiness=0))
#
#
#     def _Conv(self, inplanes, outplanes):
#         return scn.Sequential(conv3x3_sparse(inplanes, outplanes),
#                         scn.BatchNormalization(outplanes, leakiness=0))
#
#
#
#
#     def forward(self, x, labels=None, crit=1.0, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [label5, label4, label3, label2, label1] = labels
#
#         #dense
#         # out5 = self.sigmoid(self.out5_conv(in4)) # 6x20
#         # disp5, mask5 = torch.unbind(out5, 1)
#         # disp5 = disp5.unsqueeze(1)
#         # mask5 = mask5.unsqueeze(1)
#
#         if labels is None:
#             # non differentiable: impossible lors de l'entrainement
#             in4 = in4 * torch.round(mask5)
#         else:
#             in4 = in4 * label5
#
#         x = self.deconv4_0(in4)
#         x = self.add([x, in3])
#         x = self.deconv4_1(x)
#
#
#         out4 = self.sigmoid(self.sparse_to_dense(self.out4_conv(x))) # 12x40
#         # print(f"Q4: {time.time() - tic}s")
#         # tic = time.time()
#         disp4, mask4 = torch.unbind(out4, 1)
#         disp4 = disp4.unsqueeze(1)
#         mask4 = mask4.unsqueeze(1)
#
#         # tic = time.time()
#         x = self.densify4(x)
#         # print(f"densify: {time.time() - tic}s")
#         # tic = time.time()
#         if labels is None:
#             x = x * torch.round(mask4)
#         else:
#             x = x * label4
#
#         x = self.deconv3_0(x)
#         x = self.add([x, in2])
#         x = self.deconv3_1(x)
#
#         # print(f"2nd layer: {time.time() - tic}s")
#         # tic = time.time()
#
#         out3 = self.sigmoid(self.sparse_to_dense(self.out3_conv(x))) # 24x80
#         # print(f"Q3: {time.time() - tic}s")
#         # tic = time.time()
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = disp3.unsqueeze(1)
#         mask3 = mask3.unsqueeze(1)
#
#         x = self.densify3(x)
#         if labels is None:
#             x = x * torch.round(mask3)
#         else:
#             x = x * label3
#
#         x = self.deconv2_0(x)
#         x = self.add([x, in1])
#         x = self.deconv2_1(x)
#
#         # print(f"3rd layer: {time.time() - tic}s")
#         # tic = time.time()
#
#         out2 = self.sigmoid(self.sparse_to_dense(self.out2_conv(x))) # 48x160
#         # print(f"Q2: {time.time() - tic}s")
#         # tic = time.time()
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = disp2.unsqueeze(1)
#         mask2 = mask2.unsqueeze(1)
#
#         x = self.densify2(x)
#         if labels is None:
#             x = x * torch.round(mask2)
#         else:
#             x = x * label2
#
#         x = self.deconv1_0(x)
#         x = self.add([x, in0])
#         x = self.deconv1_1(x)
#
#         # print(f"4th layer: {time.time() - tic}s")
#         # tic = time.time()
#
#         out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x))) # 96x320
#         # print(f"Q1: {time.time() - tic}s")
#         # tic = time.time()
#         disp1, mask1 = torch.unbind(out1, 1)
#         disp1 = disp1.unsqueeze(1)
#         mask1 = mask1.unsqueeze(1)
#
#         x = self.densify1(x)
#         if labels is None:
#             x = x * torch.round(mask1)
#         else:
#             x = x * label1
#
#         x = self.deconv0_0(x)
#         x = self.deconv0_1(x)
#         # print(f"5th layer: {time.time() - tic}s")
#         # tic = time.time()
#
#         disp0 = self.sigmoid(self.densify_out(self.out0_conv(x))) # 192x640
#         # print(f"Q0: {time.time() - tic}s")
#
#
#
#         return [disp5, disp4, disp3, disp2, disp1, disp0], [mask5, mask4, mask3, mask2, mask1]
#
#
#
#
# class ResNet18TransposeSparseAutoMasking(nn.Module):
#
#     def __init__(self, transblock, layers, num_classes=1):
#         self.inplanes = 512
#         super(ResNet18TransposeSparseAutoMasking, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode='nearest')
#
#         # self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#         # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
#
#         self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
#         self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
#         self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
#         self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)
#
#         # self.skip0 = self._make_skip_layer(128, 64 * transblock.expansion)
#         # self.skip1 = self._make_skip_layer(256, 64 * transblock.expansion)
#         # self.skip2 = self._make_skip_layer(512, 128 * transblock.expansion)
#         # self.skip3 = self._make_skip_layer(1024, 256 * transblock.expansion)
#         # self.skip4 = self._make_skip_layer(2048, 512 * transblock.expansion)
#
#         self.skip0 = self._make_skip_layer(64, 64 * transblock.expansion)
#         self.skip1 = self._make_skip_layer(64, 64 * transblock.expansion)
#         self.skip2 = self._make_skip_layer(128, 128 * transblock.expansion)
#         self.skip3 = self._make_skip_layer(256, 256 * transblock.expansion)
#         self.skip4 = self._make_skip_layer(512, 512 * transblock.expansion)
#
#         self.densify0 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify1 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify2 = scn.SparseToDense(2, 128 * transblock.expansion)
#         self.densify3 = scn.SparseToDense(2, 256 * transblock.expansion)
#
#         self.inplanes = 64
#         self.final_deconv = self._make_transpose(transblock, 32 * transblock.expansion, 3, stride=2)
#
#         # self.final_deconv = scn.Sequential(
#         #         # scn.SparseToDense(2, 32 * transblock.expansion),
#         #         # scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
#         #         scn.SparseToDense(2, self.inplanes * transblock.expansion),
#         #         nn.ConvTranspose2d(self.inplanes * transblock.expansion, 1, kernel_size=2,
#         #                                        stride=2, padding=0, bias=True)
#         #     )
#
#         self.out6_conv = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=True)
#         self.out5_conv = scn.NetworkInNetwork(256 * transblock.expansion, 2, True)
#         self.out4_conv = scn.NetworkInNetwork(128 * transblock.expansion, 2, True)
#         self.out3_conv = scn.NetworkInNetwork(64 * transblock.expansion, 2, True)
#         self.out2_conv = scn.NetworkInNetwork(64 * transblock.expansion, 2, True)
#         self.out1_conv = scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
#
#         self.sparse_to_dense = scn.SparseToDense(2, num_classes)
#         self.sparse_to_dense_2 = scn.SparseToDense(2, 2)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU()
#         self.round = roundGrad.apply
#
#     def _make_transpose(self, transblock, planes, blocks, stride=1):
#
#         upsample = None
#         if stride != 1:
#             upsample = scn.Sequential(
#                 scn.SparseToDense(2,self.inplanes * transblock.expansion),
#                 nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
#                                   kernel_size=2, stride=stride, padding=0, bias=False),
#                 scn.DenseToSparse(2),
#                 scn.BatchNormalization(planes)
#             )
#         elif self.inplanes * transblock.expansion != planes:
#             upsample = scn.Sequential(
#                 scn.NetworkInNetwork(self.inplanes * transblock.expansion, planes, False),
#                 scn.BatchNormalization(planes)
#             )
#
#         layers = []
#
#         for i in range(1, blocks):
#             layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))
#
#         layers.append(transblock(self.inplanes, planes, stride, upsample))
#         self.inplanes = planes // transblock.expansion
#
#         return scn.Sequential(*layers)
#
#     def _make_skip_layer(self, inplanes, planes):
#
#         layers = scn.Sequential(
#             scn.NetworkInNetwork(inplanes, planes, False),
#             scn.BatchNormReLU(planes)
#         )
#         return layers
#
#     def forward(self, x, labels=None, crit=1.0, sparse_mode=True, use_skip=True):
#         use_skip=False
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [label6, label5, label4, label3, label2] = labels
#
#         out6 = self.out6_conv(in4)
#         disp6, mask6 = torch.unbind(out6, 1)
#         disp6 = self.sigmoid(disp6.unsqueeze(1))
#         mask6 = self.sigmoid(mask6.unsqueeze(1))
#
#         if labels is None:
#             in4 = in4 * self.round(mask6)
#         else:
#             in4 = in4 * label6
#         # in4 = in4 * torch.round(mask6)
#         # in4 = in4 * (mask6 > 0).type(mask6.dtype)
#
#         in4 = self.dense_to_sparse(in4)
#         # skip4 = self.skip4(in4)
#
#         x = self.deconv1(in4)
#         print(self.densify3(x).size())
#         out5 = self.sparse_to_dense_2(self.out5_conv(x))
#         disp5, mask5 = torch.unbind(out5, 1)
#         disp5 = self.sigmoid(disp5.unsqueeze(1))
#         mask5 = self.sigmoid(mask5.unsqueeze(1))
#
#         if labels is None:
#             in3 = in3 * self.round(mask5)
#         else:
#             in3 = in3 * label5
#         # in3 = in3 * torch.round(mask5)
#         # in3 = in3 * (mask5 > 0).type(mask5.dtype)
#
#         in3 = self.dense_to_sparse(in3)
#
#         if use_skip:
#             x = self.add([self.skip3(in3), self.densify3(x)])
#         else:
#             x = self.add([in3, self.densify3(x)])
#
#         # upsample 2
#         x = self.deconv2(x)
#         out4 = self.sparse_to_dense_2(self.out4_conv(x))
#         disp4, mask4 = torch.unbind(out4, 1)
#         disp4 = self.sigmoid(disp4.unsqueeze(1))
#         mask4 = self.sigmoid(mask4.unsqueeze(1))
#
#         if labels is None:
#             in2 = in2 * self.round(mask4)
#         else:
#             in2 = in2 * label4
#         # in2 = in2 * torch.round(mask4)
#         # in2 = in2 * (mask4 > 0).type(mask4.dtype)
#
#         in2 = self.dense_to_sparse(in2)
#
#         if use_skip:
#             x = self.add([self.skip2(in2), self.densify2(x)])
#         else:
#             x = self.add([in2, self.densify2(x)])
#
#         # upsample 3
#         x = self.deconv3(x)
#         out3 = self.sparse_to_dense_2(self.out3_conv(x))
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = self.sigmoid(disp3.unsqueeze(1))
#         mask3 = self.sigmoid(mask3.unsqueeze(1))
#
#         if labels is None:
#             in1 = in1 * self.round(mask3)
#         else:
#             in1 = in1 * label3
#         # in1 = in1 * torch.round(mask3)
#         # in1 = in1 * (mask3 > 0).type(mask3.dtype)
#
#         in1 = self.dense_to_sparse(in1)
#
#         if use_skip:
#             x = self.add([self.skip1(in1), self.densify1(x)])
#         else:
#             x = self.add([in1, self.densify1(x)])
#
#         # upsample 4
#         x = self.deconv4(x)
#         out2 = self.sparse_to_dense_2(self.out2_conv(x))
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = self.sigmoid(disp2.unsqueeze(1))
#         mask2 = self.sigmoid(mask2.unsqueeze(1))
#
#         if labels is None:
#             in0 = in0 * self.round(mask2)
#         else:
#             in0 = in0 * label2
#         # in0 = in0 * torch.round(mask2)
#         # in0 = in0 * (mask2 > 0).type(mask2.dtype)
#
#         in0 = self.dense_to_sparse(in0)
#
#         if use_skip:
#             x = self.add([self.skip0(in0), self.densify0(x)])
#         else:
#             x = self.add([in0, self.densify0(x)])
#
#         # final
#         x = self.final_deconv(x)
#         out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
#
#
#         return [disp6, disp5, disp4, disp3, disp2, out1], [mask6, mask5, mask4, mask3, mask2]
#
# class ResNet18TransposeSparseAutoMaskingV2(nn.Module):
#
#     def __init__(self, transblock, layers, num_classes=1):
#         self.inplanes = 512
#         super(ResNet18TransposeSparseAutoMaskingV2, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode='nearest')
#
#         # self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#         # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
#
#         self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
#         self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
#         self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
#         self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)
#
#         # self.skip0 = self._make_skip_layer(128, 64 * transblock.expansion)
#         # self.skip1 = self._make_skip_layer(256, 64 * transblock.expansion)
#         # self.skip2 = self._make_skip_layer(512, 128 * transblock.expansion)
#         # self.skip3 = self._make_skip_layer(1024, 256 * transblock.expansion)
#         # self.skip4 = self._make_skip_layer(2048, 512 * transblock.expansion)
#
#         self.skip0 = self._make_skip_layer(64, 64 * transblock.expansion)
#         self.skip1 = self._make_skip_layer(64, 64 * transblock.expansion)
#         self.skip2 = self._make_skip_layer(128, 128 * transblock.expansion)
#         self.skip3 = self._make_skip_layer(256, 256 * transblock.expansion)
#         self.skip4 = self._make_skip_layer(512, 512 * transblock.expansion)
#
#         self.densify0 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify1 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify2 = scn.SparseToDense(2, 128 * transblock.expansion)
#         self.densify3 = scn.SparseToDense(2, 256 * transblock.expansion)
#
#         self.inplanes = 64
#         self.final_deconv = self._make_transpose(transblock, 32 * transblock.expansion, 3, stride=2)
#
#         # self.final_deconv = scn.Sequential(
#         #         # scn.SparseToDense(2, 32 * transblock.expansion),
#         #         # scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
#         #         scn.SparseToDense(2, self.inplanes * transblock.expansion),
#         #         nn.ConvTranspose2d(self.inplanes * transblock.expansion, 1, kernel_size=2,
#         #                                        stride=2, padding=0, bias=True)
#         #     )
#
#         self.out6_conv = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=True)
#         self.out5_conv = scn.NetworkInNetwork(256 * transblock.expansion, 2, True)
#         self.out4_conv = scn.NetworkInNetwork(128 * transblock.expansion, 2, True)
#         self.out3_conv = scn.NetworkInNetwork(64 * transblock.expansion, 2, True)
#         self.out2_conv = scn.NetworkInNetwork(64 * transblock.expansion, 2, True)
#         self.out1_conv = scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
#
#         self.sparse_to_dense = scn.SparseToDense(2, num_classes)
#         self.sparse_to_dense_2 = scn.SparseToDense(2, 2)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU()
#         self.round = roundGrad.apply
#
#     def _make_transpose(self, transblock, planes, blocks, stride=1):
#
#         upsample = None
#         if stride != 1:
#             upsample = scn.Sequential(
#                 scn.SparseToDense(2,self.inplanes * transblock.expansion),
#                 nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
#                                   kernel_size=2, stride=stride, padding=0, bias=False),
#                 scn.DenseToSparse(2),
#                 scn.BatchNormalization(planes)
#             )
#         elif self.inplanes * transblock.expansion != planes:
#             upsample = scn.Sequential(
#                 scn.NetworkInNetwork(self.inplanes * transblock.expansion, planes, False),
#                 scn.BatchNormalization(planes)
#             )
#
#         layers = []
#
#         for i in range(1, blocks):
#             layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))
#
#         layers.append(transblock(self.inplanes, planes, stride, upsample))
#         self.inplanes = planes // transblock.expansion
#
#         return scn.Sequential(*layers)
#
#     def _make_skip_layer(self, inplanes, planes):
#
#         layers = scn.Sequential(
#             scn.NetworkInNetwork(inplanes, planes, False),
#             scn.BatchNormReLU(planes)
#         )
#         return layers
#
#     def forward(self, x, labels=None, crit=1.0, sparse_mode=True, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         # if labels is not None:
#         #     [mask4, mask3, mask2, mask1, mask0] = labels
#
#         out6 = self.out6_conv(in4)
#         disp6, mask6 = torch.unbind(out6, 1)
#         disp6 = self.sigmoid(disp6.unsqueeze(1))
#         mask6 = self.sigmoid(mask6.unsqueeze(1))
#
#         in4 = in4 * self.round(mask6)
#         # in4 = in4 * torch.round(mask6)
#         # in4 = in4 * (mask6 > 0).type(mask6.dtype)
#
#         in4 = self.dense_to_sparse(in4)
#         # skip4 = self.skip4(in4)
#
#         x = self.add([in4, disp6])
#
#         x = self.deconv1(in4)
#         out5 = self.sparse_to_dense_2(self.out5_conv(x))
#         disp5, mask5 = torch.unbind(out5, 1)
#         disp5 = self.sigmoid(disp5.unsqueeze(1))
#         mask5 = self.sigmoid(mask5.unsqueeze(1))
#
#         in3 = in3 * self.round(mask5)
#         # in3 = in3 * torch.round(mask5)
#         # in3 = in3 * (mask5 > 0).type(mask5.dtype)
#
#         in3 = self.dense_to_sparse(in3)
#
#         x = self.add([in3, self.densify3(x)])
#         x = self.add([x, disp5])
#
#         # upsample 2
#         x = self.deconv2(x)
#         out4 = self.sparse_to_dense_2(self.out4_conv(x))
#         disp4, mask4 = torch.unbind(out4, 1)
#         disp4 = self.sigmoid(disp4.unsqueeze(1))
#         mask4 = self.sigmoid(mask4.unsqueeze(1))
#
#         in2 = in2 * self.round(mask4)
#         # in2 = in2 * torch.round(mask4)
#         # in2 = in2 * (mask4 > 0).type(mask4.dtype)
#
#         in2 = self.dense_to_sparse(in2)
#
#         x = self.add([in2, self.densify2(x)])
#         x = self.add([x, disp4])
#
#         # upsample 3
#         x = self.deconv3(x)
#         out3 = self.sparse_to_dense_2(self.out3_conv(x))
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = self.sigmoid(disp3.unsqueeze(1))
#         mask3 = self.sigmoid(mask3.unsqueeze(1))
#
#         in1 = in1 * self.round(mask3)
#         # in1 = in1 * torch.round(mask3)
#         # in1 = in1 * (mask3 > 0).type(mask3.dtype)
#
#         in1 = self.dense_to_sparse(in1)
#
#         x = self.add([in1, self.densify1(x)])
#         x = self.add([x, disp3])
#
#         # upsample 4
#         x = self.deconv4(x)
#         out2 = self.sparse_to_dense_2(self.out2_conv(x))
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = self.sigmoid(disp2.unsqueeze(1))
#         mask2 = self.sigmoid(mask2.unsqueeze(1))
#
#         in0 = in0 * self.round(mask2)
#         # in0 = in0 * torch.round(mask2)
#         # in0 = in0 * (mask2 > 0).type(mask2.dtype)
#
#         in0 = self.dense_to_sparse(in0)
#
#         x = self.add([in0, self.densify0(x)])
#         x = self.add([x, disp2])
#
#         # final
#         x = self.final_deconv(x)
#         out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
#
#
#         return [disp6, disp5, disp4, disp3, disp2, out1], [mask6, mask5, mask4, mask3, mask2]
#
#
# class ResNet18TransposeSparseAutoMaskingSelf(nn.Module):
#
#     def __init__(self, transblock, layers, num_classes=1):
#         self.inplanes = 512
#         super(ResNet18TransposeSparseAutoMaskingSelf, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode='nearest')
#
#         # self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#         # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
#
#         self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
#         self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
#         self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
#         self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)
#
#         self.densify0 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify1 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify2 = scn.SparseToDense(2, 128 * transblock.expansion)
#         self.densify3 = scn.SparseToDense(2, 256 * transblock.expansion)
#
#         self.inplanes = 64
#         self.final_deconv = self._make_transpose(transblock, 32 * transblock.expansion, 3, stride=2)
#
#         self.out6_conv = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=True)
#         self.out5_conv = scn.NetworkInNetwork(256 * transblock.expansion, 2, True)
#         self.out4_conv = scn.NetworkInNetwork(128 * transblock.expansion, 2, True)
#         self.out3_conv = scn.NetworkInNetwork(64 * transblock.expansion, 2, True)
#         self.out2_conv = scn.NetworkInNetwork(64 * transblock.expansion, 2, True)
#         self.out1_conv = scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
#
#         self.sparse_to_dense = scn.SparseToDense(2, num_classes)
#         self.sparse_to_dense_2 = scn.SparseToDense(2, 2)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU()
#
#     def _make_transpose(self, transblock, planes, blocks, stride=1):
#
#         upsample = None
#         if stride != 1:
#             upsample = scn.Sequential(
#                 scn.SparseToDense(2,self.inplanes * transblock.expansion),
#                 nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
#                                   kernel_size=2, stride=stride, padding=0, bias=False),
#                 scn.DenseToSparse(2),
#                 scn.BatchNormalization(planes)
#             )
#         elif self.inplanes * transblock.expansion != planes:
#             upsample = scn.Sequential(
#                 scn.NetworkInNetwork(self.inplanes * transblock.expansion, planes, False),
#                 scn.BatchNormalization(planes)
#             )
#
#         layers = []
#
#         for i in range(1, blocks):
#             layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))
#
#         layers.append(transblock(self.inplanes, planes, stride, upsample))
#         self.inplanes = planes // transblock.expansion
#
#         return scn.Sequential(*layers)
#
#     def forward(self, x, labels=None, crit=1.0, sparse_mode=True, use_skip=True, training=True):
#         [in0, in1, in2, in3, in4] = x
#
#         out6 = self.sigmoid(self.out6_conv(in4))
#         disp6, mask6 = torch.unbind(out6, 1)
#         disp6 = disp6.unsqueeze(1)
#         mask6 = mask6.unsqueeze(1)
#
#         if not training:
#             in4 = in4 * torch.round(mask6)
#
#         in4 = self.dense_to_sparse(in4)
#
#         x = self.deconv1(in4)
#         out5 = self.sigmoid(self.sparse_to_dense_2(self.out5_conv(x)))
#         disp5, mask5 = torch.unbind(out5, 1)
#         disp5 = disp5.unsqueeze(1)
#         mask5 = mask5.unsqueeze(1)
#
#         if not training:
#             in3 = in3 * torch.round(mask5)
#         else:
#             in3 = in3 * torch.round(self.up(mask6))
#
#         in3 = self.dense_to_sparse(in3)
#
#         if use_skip:
#             x = self.add([in3,self.densify3(x)])
#
#         # upsample 2
#         x = self.deconv2(x)
#         out4 = self.sigmoid(self.sparse_to_dense_2(self.out4_conv(x)))
#         disp4, mask4 = torch.unbind(out4, 1)
#         disp4 = disp4.unsqueeze(1)
#         mask4 = mask4.unsqueeze(1)
#
#         if not training:
#             in2 = in2 * torch.round(mask4)
#         else:
#             in2 = in2 * torch.round(self.up(mask5))
#
#         in2 = self.dense_to_sparse(in2)
#
#         if use_skip:
#             x = self.add([in2,self.densify2(x)])
#
#         # upsample 3
#         x = self.deconv3(x)
#         out3 = self.sigmoid(self.sparse_to_dense_2(self.out3_conv(x)))
#         disp3, mask3 = torch.unbind(out3, 1)
#         disp3 = disp3.unsqueeze(1)
#         mask3 = mask3.unsqueeze(1)
#
#         if not training:
#             in1 = in1 * torch.round(mask3)
#         else:
#             in1 = in1 * torch.round(self.up(mask4))
#
#         in1 = self.dense_to_sparse(in1)
#
#         if use_skip:
#             x = self.add([in1,self.densify1(x)])
#
#         # upsample 4
#         x = self.deconv4(x)
#         out2 = self.sigmoid(self.sparse_to_dense_2(self.out2_conv(x)))
#         disp2, mask2 = torch.unbind(out2, 1)
#         disp2 = disp2.unsqueeze(1)
#         mask2 = mask2.unsqueeze(1)
#
#         if not training:
#             in0 = in0 * torch.round(mask2)
#         else:
#             in0 = in0 * torch.round(self.up(mask3))
#
#         in0 = self.dense_to_sparse(in0)
#
#         if use_skip:
#             x = self.add([in0, self.densify0(x)])
#
#         # final
#         x = self.final_deconv(x)
#         out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
#
#
#         return [disp6, disp5, disp4, disp3, disp2, out1], [mask6, mask5, mask4, mask3, mask2]
#
#
# # class ResNet18TransposeSparseAutoMaskingV2(nn.Module):
# #
# #     def __init__(self, transblock, layers, num_classes=1):
# #         self.inplanes = 512
# #         super(ResNet18TransposeSparseAutoMaskingV2, self).__init__()
# #
# #         self.dense_to_sparse = scn.DenseToSparse(2)
# #         self.add = AddSparseDense()
# #         self.up = nn.Upsample(scale_factor=2, mode='nearest')
# #
# #         # self.num_ch_enc = np.array([64, 64, 128, 256, 512])
# #         # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
# #
# #         self.deconv1 = self._make_transpose(transblock, 256 * transblock.expansion, layers[0], stride=2)
# #         self.deconv2 = self._make_transpose(transblock, 128 * transblock.expansion, layers[1], stride=2)
# #         self.deconv3 = self._make_transpose(transblock, 64 * transblock.expansion, layers[2], stride=2)
# #         self.deconv4 = self._make_transpose(transblock, 64 * transblock.expansion, layers[3], stride=2)
# #
# #         self.densify0 = scn.SparseToDense(2, 64 * transblock.expansion)
# #         self.densify1 = scn.SparseToDense(2, 64 * transblock.expansion)
# #         self.densify2 = scn.SparseToDense(2, 128 * transblock.expansion)
# #         self.densify3 = scn.SparseToDense(2, 256 * transblock.expansion)
# #
# #         self.inplanes = 64
# #         self.final_deconv = self._make_transpose(transblock, 32 * transblock.expansion, 3, stride=2)
# #
# #         self.out6_conv = nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=True)
# #         self.out5_conv = scn.NetworkInNetwork(256 * transblock.expansion, 1, True)
# #         self.out4_conv = scn.NetworkInNetwork(128 * transblock.expansion, 1, True)
# #         self.out3_conv = scn.NetworkInNetwork(64 * transblock.expansion, 1, True)
# #         self.out2_conv = scn.NetworkInNetwork(64 * transblock.expansion, 1, True)
# #         self.out1_conv = scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
# #
# #
# #         self.mask6_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=True)
# #         self.mask5_conv = scn.NetworkInNetwork(1, 1, True)
# #         self.mask4_conv = scn.NetworkInNetwork(1, 1, True)
# #         self.mask3_conv = scn.NetworkInNetwork(1, 1, True)
# #         self.mask2_conv = scn.NetworkInNetwork(1, 1, True)
# #         self.mask1_conv = scn.NetworkInNetwork(1, 1, True)
# #
# #
# #         self.sparse_to_dense = scn.SparseToDense(2, num_classes)
# #         self.sparse_to_dense_2 = scn.SparseToDense(2, 2)
# #         self.sigmoid = nn.Sigmoid()
# #         self.relu = nn.ReLU()
# #
# #     def _make_transpose(self, transblock, planes, blocks, stride=1):
# #
# #         upsample = None
# #         if stride != 1:
# #             upsample = scn.Sequential(
# #                 scn.SparseToDense(2,self.inplanes * transblock.expansion),
# #                 nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
# #                                   kernel_size=2, stride=stride, padding=0, bias=False),
# #                 scn.DenseToSparse(2),
# #                 scn.BatchNormalization(planes)
# #             )
# #         elif self.inplanes * transblock.expansion != planes:
# #             upsample = scn.Sequential(
# #                 scn.NetworkInNetwork(self.inplanes * transblock.expansion, planes, False),
# #                 scn.BatchNormalization(planes)
# #             )
# #
# #         layers = []
# #
# #         for i in range(1, blocks):
# #             layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))
# #
# #         layers.append(transblock(self.inplanes, planes, stride, upsample))
# #         self.inplanes = planes // transblock.expansion
# #
# #         return scn.Sequential(*layers)
# #
# #
# #     def forward(self, x, labels=None, crit=1.0, sparse_mode=True, use_skip=True):
# #         [in0, in1, in2, in3, in4] = x
# #
# #         if labels is not None:
# #             [mask4, mask3, mask2, mask1, mask0] = labels
# #
# #         out6 = self.out6_conv(in4)
# #         disp6 = self.sigmoid(out6)
# #         mask6 = torch.round(self.sigmoid(out6))
# #
# #         in4 = in4 * mask6
# #         in4 = self.dense_to_sparse(in4)
# #
# #         x = self.deconv1(in4)
# #         out5 = self.out5_conv(x)
# #         disp5 = self.sigmoid(self.sparse_to_dense(out5))
# #         mask5 = torch.round(self.sigmoid(self.sparse_to_dense(self.mask5_conv(out5))))
# #
# #
# #         in3 = in3 * mask5
# #         in3 = self.dense_to_sparse(in3)
# #
# #         if use_skip:
# #             x = self.add([in3,self.densify3(x)])
# #
# #         # upsample 2
# #         x = self.deconv2(x)
# #         out4 = self.out4_conv(x)
# #         disp4 = self.sigmoid(self.sparse_to_dense(out4))
# #         mask4 = torch.round(self.sigmoid(self.sparse_to_dense(self.mask4_conv(out4))))
# #
# #         in2 = in2 * mask4
# #         in2 = self.dense_to_sparse(in2)
# #
# #         if use_skip:
# #             x = self.add([in2,self.densify2(x)])
# #
# #         # upsample 3
# #         x = self.deconv3(x)
# #         out3 = self.out3_conv(x)
# #         disp3 = self.sigmoid(self.sparse_to_dense(out3))
# #         mask3 = torch.round(self.sigmoid(self.sparse_to_dense(self.mask4_conv(out3))))
# #
# #         in1 = in1 * mask3
# #         in1 = self.dense_to_sparse(in1)
# #
# #         if use_skip:
# #             x = self.add([in1,self.densify1(x)])
# #
# #         # upsample 4
# #         x = self.deconv4(x)
# #         out2 = self.out2_conv(x)
# #         disp2 = self.sigmoid(self.sparse_to_dense(out2))
# #         mask2 = torch.round(self.sigmoid(self.sparse_to_dense(self.mask4_conv(out2))))
# #
# #         in0 = in0 * mask2
# #         in0 = self.dense_to_sparse(in0)
# #
# #         if use_skip:
# #             x = self.add([in0, self.densify0(x)])
# #
# #         # final
# #         x = self.final_deconv(x)
# #         out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
# #
# #
# #         return [disp6, disp5, disp4, disp3, disp2, out1], [mask6, mask5, mask4, mask3, mask2]
#
#
# class ResNet18TransposeSparseLight(nn.Module):
#
#     def __init__(self, transblock, transblock_dense, layers, num_classes=1):
#         self.inplanes = 512
#         super(ResNet18TransposeSparseLight, self).__init__()
#
#         self.dense_to_sparse = scn.DenseToSparse(2)
#         self.add = AddSparseDense()
#         self.up = nn.Upsample(scale_factor=2, mode='nearest')
#
#         # self.num_ch_enc = np.array([64, 64, 128, 256, 512])
#         # self.num_ch_dec = np.array([16, 32, 64, 128, 256])
#         self.deconv1 = self._make_transpose_dense(transblock_dense, 256 * transblock_dense.expansion, layers[0], stride=2)
#         self.deconv2 = self._make_transpose_dense(transblock_dense, 128 * transblock_dense.expansion, layers[1], stride=2)
#         self.deconv3 = self._make_transpose_sparse(transblock, 64 * transblock.expansion, layers[2], stride=2)
#         self.deconv4 = self._make_transpose_sparse(transblock, 64 * transblock.expansion, layers[3], stride=2)
#
#         # self.skip0 = self._make_skip_layer(128, 64 * transblock.expansion)
#         # self.skip1 = self._make_skip_layer(256, 64 * transblock.expansion)
#         # self.skip2 = self._make_skip_layer(512, 128 * transblock.expansion)
#         # self.skip3 = self._make_skip_layer(1024, 256 * transblock.expansion)
#         # self.skip4 = self._make_skip_layer(2048, 512 * transblock.expansion)
#
#         self.densify0 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify1 = scn.SparseToDense(2, 64 * transblock.expansion)
#         self.densify2 = scn.SparseToDense(2, 128 * transblock.expansion)
#         self.densify3 = scn.SparseToDense(2, 256 * transblock.expansion)
#
#         self.inplanes = 64
#         self.final_deconv = self._make_transpose_sparse(transblock, 32 * transblock.expansion, 3, stride=2)
#
#         # self.final_deconv = scn.Sequential(
#         #         # scn.SparseToDense(2, 32 * transblock.expansion),
#         #         # scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
#         #         scn.SparseToDense(2, self.inplanes * transblock.expansion),
#         #         nn.ConvTranspose2d(self.inplanes * transblock.expansion, 1, kernel_size=2,
#         #                                        stride=2, padding=0, bias=True)
#         #     )
#
#         # self.out6_conv = nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=True)
#         # self.out5_conv = scn.NetworkInNetwork(256 * transblock.expansion, 1, True)
#         # self.out4_conv = scn.NetworkInNetwork(128 * transblock.expansion, 1, True)
#         self.out4_conv = nn.Conv2d(128, 1, kernel_size=1, stride=1, bias=True)
#         self.out3_conv = scn.NetworkInNetwork(64 * transblock.expansion, 1, True)
#         self.out2_conv = scn.NetworkInNetwork(64 * transblock.expansion, 1, True)
#         self.out1_conv = scn.NetworkInNetwork(32 * transblock.expansion, 1, True)
#
#         self.sparse_to_dense = scn.SparseToDense(2, num_classes)
#         self.sigmoid = nn.Sigmoid()
#
#     def _make_transpose_dense(self, transblock, planes, blocks, stride=1):
#
#         upsample = None
#         if stride != 1:
#             upsample = nn.Sequential(
#                 nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
#                                    kernel_size=2, stride=stride,
#                                    padding=0, bias=False),
#                 SynchronizedBatchNorm2d(planes),
#             )
#         elif self.inplanes * transblock.expansion != planes:
#             upsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes * transblock.expansion, planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 SynchronizedBatchNorm2d(planes),
#             )
#
#         layers = []
#
#         for i in range(1, blocks):
#             layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))
#
#         layers.append(transblock(self.inplanes, planes, stride, upsample))
#         self.inplanes = planes // transblock.expansion
#
#         return nn.Sequential(*layers)
#
#     def _make_transpose_sparse(self, transblock, planes, blocks, stride=1):
#
#         upsample = None
#         if stride != 1:
#             upsample = scn.Sequential(
#                 scn.SparseToDense(2,self.inplanes * transblock.expansion),
#                 nn.ConvTranspose2d(self.inplanes * transblock.expansion, planes,
#                                   kernel_size=2, stride=stride, padding=0, bias=False),
#                 scn.DenseToSparse(2),
#                 scn.BatchNormalization(planes)
#             )
#         elif self.inplanes * transblock.expansion != planes:
#             upsample = scn.Sequential(
#                 scn.NetworkInNetwork(self.inplanes * transblock.expansion, planes, False),
#                 scn.BatchNormalization(planes)
#             )
#
#         layers = []
#
#         for i in range(1, blocks):
#             layers.append(transblock(self.inplanes, self.inplanes * transblock.expansion))
#
#         layers.append(transblock(self.inplanes, planes, stride, upsample))
#         self.inplanes = planes // transblock.expansion
#
#         return scn.Sequential(*layers)
#
#     def _make_skip_layer(self, inplanes, planes):
#
#         layers = scn.Sequential(
#             scn.NetworkInNetwork(inplanes, planes, False),
#             scn.BatchNormReLU(planes)
#         )
#         return layers
#
#     def _masking(self, out, crit=0.5):
#         out = 1/80 + (1/0.1 - 1/80) * out
#         a = out[:,:,0::2,0::2]
#         b = out[:,:,0::2,1::2]
#         c = out[:,:,1::2,0::2]
#         d = out[:,:,1::2,1::2]
#
#         m_max = torch.max(torch.max(torch.max(a,b),c),d)
#         m_min = torch.min(torch.min(torch.min(a,b),c),d)
#
#         mask = self.up(m_max - m_min) > crit
#
#         return mask.type(out.dtype)
#
#
#     def forward(self, x, labels=None, crit=1.0, sparse_mode=True, use_skip=True):
#         [in0, in1, in2, in3, in4] = x
#
#         if labels is not None:
#             [mask3, mask2, mask1, mask0] = labels
#
#         # out6 = self.sigmoid(self.out6_conv(in4))
#
#         # if labels is None:
#         #     # mask4 = self._masking(out6, crit)
#         #     mask4 = torch.ones_like(out6) # on force la non segmentation
#         #     if torch.all(mask4 == torch.zeros_like(mask4)):
#         #         mask4 = (torch.rand_like(mask4) > 0.4).type(mask4.dtype)
#         # in4 = in4 * mask4
#
#         # in4 = self.dense_to_sparse(in4)
#         # skip4 = self.skip4(in4)
#         # upsample 1
#
#         x = self.deconv1(in4)
#         # out5 = self.sigmoid(self.sparse_to_dense(self.out5_conv(x)))
#
#
#         # if labels is None:
#         #     # mask3 = self.up(mask4) * self._masking(out5, crit)
#         #     mask3 = torch.ones_like(out5)
#         #     if torch.all(mask3 == torch.zeros_like(mask3)):
#         #         mask3 = self.up(mask4 * (torch.rand_like(mask4) > 0.4).type(mask4.dtype))
#         # in3 = in3 * mask3
#         #
#         # in3 = self.dense_to_sparse(in3)
#
#         if use_skip:
#             x = x + in3 #self.add([in3,x])
#             # x = self.add([in3,self.densify3(x)])
#             # x = self.add([self.skip3(in3),self.densify3(x)])
#
#         # upsample 2
#         x = self.deconv2(x)
#         out4 = self.sigmoid(self.out4_conv(x))
#
#         if labels is None:
#             # mask2 = self.up(mask3) * self._masking(out4, crit)
#             mask2 = self._masking(out4, crit)
#             if torch.all(mask2 == 0):#torch.zeros_like(mask2)):
#                 mask2 = torch.round(torch.rand_like(mask2))
#         in2 = in2 * mask2
#
#         in2 = self.dense_to_sparse(in2)
#
#         if use_skip:
#             # x = self.add([self.skip2(in2),self.densify2(x)])
#             # x = self.add([in2,self.densify2(x)])
#             x = self.add([in2,x])
#
#         # upsample 3
#         x = self.deconv3(x)
#         out3 = self.sigmoid(self.sparse_to_dense(self.out3_conv(x)))
#
#         if labels is None:
#             mask1 = self.up(mask2) * self._masking(out3, crit)
#             if torch.all(mask1 == 0):#torch.zeros_like(mask1)):
#                 mask1 = self.up(mask2) * torch.round(torch.rand_like(mask1))
#         in1 = in1 * mask1
#
#         in1 = self.dense_to_sparse(in1)
#
#         if use_skip:
#             x = self.add([in1,self.densify1(x)])
#
#         # upsample 4
#         x = self.deconv4(x)
#         out2 = self.sigmoid(self.sparse_to_dense(self.out2_conv(x)))
#
#         if labels is None:
#             mask0 = self.up(mask1) * self._masking(out2, crit)
#             if torch.all(mask0 == 0):#torch.zeros_like(mask0)):
#                 mask0 = self.up(mask1) * torch.round(torch.rand_like(mask0))
#         in0 = in0 * mask0
#
#         in0 = self.dense_to_sparse(in0)
#
#         if use_skip:
#             # x = self.add([in0, x])
#             x = self.add([in0, self.densify0(x)])
#
#         # final
#         # x = self.final_conv(x)
#         # out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
#         # out1 = self.sigmoid(self.final_deconv(x))
#         x = self.final_deconv(x)
#         out1 = self.sigmoid(self.sparse_to_dense(self.out1_conv(x)))
#         # out1 = self.sigmoid(self.final_deconv(x))
#
#
#         return [out4, out3, out2, out1], [mask2, mask1, mask0]
#
# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on Places
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     # if pretrained:
#     #     model.load_state_dict(load_url(model_urls['resnet18']), strict=False)
#     return model
#
# def resnet50(pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on Places
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
#     return model
#
#
# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on Places
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
#     return model
#
# def resnetSparse18(pretrained=False, **kwargs):
#     """Constructs a ResNet-Sparse-18 transpose model.
#     """
#     model = ResNetSparse(BasicBlockSparse, [2, 2, 2, 2], **kwargs)
#     return model
#
#
# def resnet34_transpose(**kwargs):
#     """Constructs a ResNet-34 transpose model.
#     """
#     model = ResNetTranspose(TransBasicBlock, [6, 4, 3, 3], **kwargs)
#     return model
#
#
# def resnet50_transpose(**kwargs):
#     """Constructs a ResNet-50 transpose model.
#     """
#     model = ResNetTranspose(TransBottleneck, [6, 4, 3, 3], **kwargs)
#     return model
#
#
# def resnet101_transpose(**kwargs):
#     """Constructs a ResNet-101 transpose model.
#     """
#     model = ResNetTranspose(TransBottleneck, [23, 4, 3, 3], **kwargs)
#     return model
#
#
# def resnet34_transpose_sparse(**kwargs):
#     """Constructs a ResNet-34 transpose model.
#     """
#     model = ResNetTransposeSparse(TransBasicBlockSparse, [6, 4, 3, 3], **kwargs)
#     return model
#
# def resnet18_transpose_sparse(**kwargs):
#     """Constructs a ResNet-18 transpose model.
#     """
#     model = ResNet18TransposeSparse(TransBottleneckSparse, [2, 2, 2, 2], **kwargs)
#     return model
#
# def resnet18_transport_sparse_automasking(**kwargs):
#     model = ResNet18TransposeSparseAutoMasking(TransBasicBlockSparse, [2, 2, 2, 2], **kwargs)
#     return model
#
# def resnet18_transport_sparse_light(**kwargs):
#     model = ResNet18TransposeSparseLight(TransBasicBlockSparse, TransBasicBlock, [2, 2, 2, 2], **kwargs)
#     return model
#
# def resnet18_transport_sparse_automasking_v2(**kwargs):
#     model = ResNet18TransposeSparseAutoMaskingV2(TransBasicBlockSparse, [2, 2, 2, 2], **kwargs)
#     return model
#
# def resnet18_transport_sparse_self_automasking(**kwargs):
#     model = ResNet18TransposeSparseAutoMaskingSelf(TransBasicBlockSparse, [2, 2, 2, 2], **kwargs)
#     return model
#
#
#
#
#
# def resnet50_transpose_sparse(**kwargs):
#     """Constructs a ResNet-50 transpose model.
#     """
#     model = ResNetTransposeSparse(TransBottleneckSparse, [6, 4, 3, 3], **kwargs)
#     return model
#
#
# def resnet101_transpose_sparse(**kwargs):
#     """Constructs a ResNet-101 transpose model.
#     """
#     model = ResNetTransposeSparse(TransBottleneckSparse, [23, 4, 3, 3], **kwargs)
#     return model
#
#
# def load_url(url, model_dir='./pretrained', map_location=None):
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     filename = url.split('/')[-1]
#     cached_file = os.path.join(model_dir, filename)
#     if not os.path.exists(cached_file):
#         sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
#         urlretrieve(url, cached_file)
#     return torch.load(cached_file, map_location=map_location)

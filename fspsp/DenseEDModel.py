import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import numpy as np
from torch.nn import functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False, dilated_bool=False):
        super(_DenseLayer, self).__init__()
        if dilated_bool:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                               kernel_size=1, stride=1, bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                               kernel_size=3, stride=1, padding=2, dilation=2, bias=False)),
            self.drop_rate = drop_rate
            self.efficient = efficient
        else:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                            kernel_size=1, stride=1, bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False)),
            self.drop_rate = drop_rate
            self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride_bool=False):
        super(_Transition, self).__init__()
        if stride_bool:
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AdaptiveAvgPool2d((32, 32)))
        else:
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False, dilated_bool=False):
        super(_DenseBlock, self).__init__()
        if dilated_bool:
            for i in range(num_layers):
                # if i == 0:
                #     layer = _DenseLayer(
                #         num_input_features + i * growth_rate,
                #         growth_rate=growth_rate,
                #         bn_size=bn_size,
                #         drop_rate=drop_rate,
                #         efficient=efficient,
                #         dilated_bool=True
                #     )
                # else:
                #     layer = _DenseLayer(
                #         num_input_features + i * growth_rate,
                #         growth_rate=growth_rate,
                #         bn_size=bn_size,
                #         drop_rate=drop_rate,
                #         efficient=efficient,
                #         dilated_bool=False
                #     )
                # self.add_module('denselayer%d' % (i + 1), layer)
                layer = _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    efficient=efficient,
                    dilated_bool=True
                )
                self.add_module('denselayer%d' % (i + 1), layer)
        else:
            for i in range(num_layers):
                layer = _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    efficient=efficient,
                    dilated_bool=False
                )
                self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseEDNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24), compression=0.5,
                 num_init_features=96, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=False, efficient=False, backbone_grad=True,
                 deconv_fine_grad=True, deconv_coarse_grad=True):

        super(DenseEDNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.dense = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.dense = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.dense.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.dense.add_module('relu0', nn.ReLU(inplace=True))
            self.dense.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
                dilated_bool=False

            )
            self.dense.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression),
                                    stride_bool=False
                                    )

                self.dense.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # # Final batch norm
        # self.dense.add_module('norm5', nn.BatchNorm2d(num_features))

        # fix backbone selectively
        # for param in self.parameters():
        #     param.requires_grad = backbone_grad

        # Decoding Block
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(2208+1056+128, 848, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(528+384+848+128, 472, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(264+192+192+472+128, 312, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        # self.deconv_readout = nn.Sequential(
        #     nn.Conv2d(132+96+96+270, 256, 3, padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(256, 96, 3, padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(96, 3, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(3, 1, 1),
        #     nn.Sigmoid()
        # )

        # inception v1
        # self.deconv_inc1 = nn.Sequential(
        #     nn.Conv2d(132 + 96 + 96 + 270, 256, 3, padding=(1, 1)),
        #     nn.ReLU(inplace=True)
        # )
        # self.deconv_inc2 = nn.Sequential(
        #     nn.Conv2d(256, 96, 3, padding=(1, 1), dilation=1),
        #     nn.ReLU(inplace=True)
        # )
        # self.deconv_inc3 = nn.Sequential(
        #     nn.Conv2d(256, 96, 3, padding=(4, 4), dilation=4),
        #     nn.ReLU(inplace=True)
        # )
        # self.deconv_inc4 = nn.Sequential(
        #     nn.Conv2d(256, 96, 3, padding=(8, 8), dilation=8),
        #     nn.ReLU(inplace=True)
        # )
        # self.deconv_readout = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(288, 96, 1, padding=(0, 0)),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(96, 3, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3)),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(3, 1, 1),
        #     nn.Sigmoid()
        # )

        # 加入inception m v2
        self.deconv_inc1 = nn.Sequential(
            nn.Conv2d(312, 96, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),

        )
        self.deconv_inc2 = nn.Sequential(
            nn.ConvTranspose2d(96, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.deconv_inc3 = nn.Sequential(
            # nn.ConvTranspose2d(96, 3, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3)),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        # self.deconv_inc4 = nn.Sequential(
        #     # nn.ConvTranspose2d(96, 3, kernel_size=(16, 16), stride=(2, 2), padding=(7, 7)),
        #     # nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(96, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(48, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(24, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        #     nn.ReLU(inplace=True),

        # )
        self.deconv_inc5 = nn.Sequential(
            nn.Conv2d(96, 3, 1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.deconv_inc6 = nn.Sequential(
            nn.Conv2d(96, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.deconv_inc7 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.deconv_readout = nn.Sequential(
            # nn.Conv2d(96, 3, 3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(15, 3, 1, padding=0),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(32, 3, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, 3, padding=(1, 1)),
            nn.Sigmoid()
        )

        self.scene = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2208, 128, 1),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(size=(15, 20))
            nn.UpsamplingBilinear2d(scale_factor=8)

        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.trans3_side1 = nn.Sequential(
            # nn.Conv2d(1056, 528, 3, padding=(1, 1)),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2)
            nn.ConvTranspose2d(1056, 528, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(1056, 528, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.trans3_side2 = nn.Sequential(
            # nn.Conv2d(1056, 528, 3, padding=(1, 1)),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.Conv2d(528, 264, 3, padding=(1, 1)),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2)
            # nn.ConvTranspose2d(1056, 528, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(528, 264, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(528, 264, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2)
        )
        # self.trans3_side3 = nn.Sequential(
        #     nn.Conv2d(264, 132, 3, padding=(1, 1)),
        #     nn.UpsamplingBilinear2d(scale_factor=2)
        # )
        self.trans2_side1 = nn.Sequential(
            # nn.Conv2d(384, 192, 3, padding=(1, 1)),
            # nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2)
            nn.ConvTranspose2d(384, 192, kernel_size=(4, 4), stride=2, padding=(1, 1)),
            nn.ReLU(inplace=True),
            # nn.UpsamplingBilinear2d(scale_factor=2)
        )

        # self.trans2_side2 = nn.Sequential(
        #     nn.Conv2d(192, 96, 3, padding=(1, 1)),
        #     nn.UpsamplingBilinear2d(scale_factor=2)
        # )
        # self.trans1_side = nn.Sequential(
        #     nn.Conv2d(192, 96, 3, padding=(1, 1)),
        #     nn.UpsamplingBilinear2d(scale_factor=2)
        # )


    def forward(self, x, weights = None):
        out = self.dense.conv0(x)
        # assert out.size() == (1, 96, 128, 128)
        out = self.dense.norm0(out)
        out = self.dense.relu0(out)
        out_pre = self.dense.pool0(out)
        # assert out.size() == (1, 96, 64, 64)
        block1 = self.dense.denseblock1(out_pre)
        trans1 = self.dense.transition1(block1)
        # assert trans1.size() == (1, 192, 32, 32)
        out = self.dense.denseblock2(trans1)
        trans2 = self.dense.transition2(out)
        # assert trans2.size() == (1, 384, 16, 16)
        out = self.dense.denseblock3(trans2)
        # assert out.size() == (1, 2112, 16, 16)
        trans3 = self.dense.transition3(out)
        # assert trans3.size() == (1, 1056, 8, 8)
        out = self.dense.denseblock4(trans3)

        if weights is None:
            scene = self.scene(out)

            scene_side1 = self.upsample(scene)
            scene_side2 = self.upsample(scene_side1)
            trans3_side1 = self.trans3_side1(trans3)
            trans3_side2 = self.trans3_side2(trans3_side1)
            # trans3_side3 = self.trans3_side3(trans3_side2)
            trans2_side1 = self.trans2_side1(trans2)
            # trans2_side2 = self.trans2_side2(trans2_side1)
            # trans1_side = self.trans1_side(trans1)

            out = torch.cat((trans3, out, scene), 1)
            out = self.deconv_layer1(out)
            # assert out.size() == (1, 816, 16, 16)
            out = torch.cat((trans3_side1, trans2, out, scene_side1), 1)
            out = self.deconv_layer2(out)
            # assert out.size() == (1, 432, 32, 32)
            out = torch.cat((trans3_side2, trans2_side1, trans1, out, scene_side2), 1)
            out = self.deconv_layer3(out)
            # assert out.size() == (1, 270, 64, 64)
            # out = torch.cat((trans3_side3, trans2_side2, trans1_side, out), 1)
            # out = self.deconv_readout(out)
            # assert out.size() == (1, 1, 256, 256)

            out = self.deconv_inc1(out)
            inc2 = self.deconv_inc2(out)
            inc3 = self.deconv_inc3(out)
            # inc4 = self.deconv_inc4(out)
            inc5 = self.deconv_inc5(out)
            inc6 = self.deconv_inc6(out)
            inc7 = self.deconv_inc7(out)
            out = torch.cat((inc2, inc3, inc5, inc6, inc7), 1)
            out = self.deconv_readout(out)

            # out = out.squeeze(1)

        else:
            # scene
            scene = F.adaptive_avg_pool2d(out, (1, 1))
            scene = F.conv2d(scene, weights['scene.1.weight'].to(device), weights['scene.1.bias'].to(device))
            scene = F.relu(scene, inplace=True)
            # scene = F.upsample(scene, size=(9, 16), scale_factor=None, mode='bilinear', align_corners=None)
            scene = F.upsample(scene, size=None, scale_factor=8, mode='bilinear', align_corners=None)
            scene_side1 = F.upsample(scene, size=None, scale_factor=2, mode='bilinear', align_corners=None)
            scene_side2 = F.upsample(scene_side1, size=None, scale_factor=2, mode='bilinear', align_corners=None)


            trans3_side1 = F.conv_transpose2d(trans3, weights['trans3_side1.0.weight'].to(device),
                                              weights['trans3_side1.0.bias'].to(device), stride=(2, 2), padding=(1, 1))
            trans3_side1 = F.relu(trans3_side1, inplace=True)
            trans3_side2 = F.conv_transpose2d(trans3_side1, weights['trans3_side2.0.weight'].to(device),
                                              weights['trans3_side2.0.bias'].to(device), stride=(2, 2), padding=(1, 1))
            trans3_side2 = F.relu(trans3_side2, inplace=True)
            trans2_side1 = F.conv_transpose2d(trans2, weights['trans2_side1.0.weight'].to(device),
                                              weights['trans2_side1.0.bias'].to(device), stride=(2, 2), padding=(1, 1))
            trans2_side1 = F.relu(trans2_side1, inplace=True)

            out = torch.cat((trans3, out, scene), 1)
            # deconv_layer1
            out = F.conv2d(out, weights['deconv_layer1.0.weight'].to(device),
                           weights['deconv_layer1.0.bias'].to(device), padding=(1, 1))
            out = F.relu(out, inplace=True)
            out = F.upsample(out, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            out = torch.cat((trans3_side1, trans2, out, scene_side1), 1)
            # deconv_layer2
            out = F.conv2d(out, weights['deconv_layer2.0.weight'].to(device),
                           weights['deconv_layer2.0.bias'].to(device), padding=(1, 1))
            out = F.relu(out, inplace=True)
            out = F.upsample(out, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            out = torch.cat((trans3_side2, trans2_side1, trans1, out, scene_side2), 1)
            # deconv_layer2
            out = F.conv2d(out, weights['deconv_layer3.0.weight'].to(device),
                           weights['deconv_layer3.0.bias'].to(device), padding=(1, 1))
            out = F.relu(out, inplace=True)
            out = F.upsample(out, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            # deconv_inc1
            out = F.conv2d(out, weights['deconv_inc1.0.weight'].to(device), weights['deconv_inc1.0.bias'].to(device),
                           padding=(1, 1))
            out = F.relu(out, inplace=True)
            out = F.upsample(out, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            # deconv_inc2
            inc2 = F.conv_transpose2d(out, weights['deconv_inc2.0.weight'].to(device),
                                      weights['deconv_inc2.0.bias'].to(device), stride=(2, 2), padding=(1, 1))
            inc2 = F.relu(inc2, inplace=True)

            # deconv_inc3
            inc3 = F.conv_transpose2d(out, weights['deconv_inc3.0.weight'].to(device),
                                      weights['deconv_inc3.0.bias'].to(device), stride=(1, 1), padding=(1, 1))
            inc3 = F.relu(inc3, inplace=True)
            inc3 = F.conv_transpose2d(inc3, weights['deconv_inc3.2.weight'].to(device),
                                      weights['deconv_inc3.2.bias'].to(device), stride=(2, 2), padding=(1, 1))
            inc3 = F.relu(inc3, inplace=True)

            # deconv_inc5
            inc5 = F.conv2d(out, weights['deconv_inc5.0.weight'].to(device), weights['deconv_inc5.0.bias'].to(device))
            inc5 = F.relu(inc5, inplace=True)
            inc5 = F.upsample(inc5, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            # deconv_inc6
            inc6 = F.conv2d(out, weights['deconv_inc6.0.weight'].to(device), weights['deconv_inc6.0.bias'].to(device),
                            padding=1)
            inc6 = F.relu(inc6, inplace=True)
            inc6 = F.upsample(inc6, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            # deconv_inc7
            inc7 = F.conv2d(out, weights['deconv_inc7.0.weight'].to(device), weights['deconv_inc7.0.bias'].to(device),
                            padding=1)
            inc7 = F.relu(inc7, inplace=True)
            inc7 = F.conv2d(inc7, weights['deconv_inc7.2.weight'].to(device), weights['deconv_inc7.2.bias'].to(device),
                            padding=1)
            inc7 = F.relu(inc7, inplace=True)
            inc7 = F.upsample(inc7, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            out = torch.cat((inc2, inc3, inc5, inc6, inc7), 1)

            # deconv_readout
            out = F.conv2d(out, weights['deconv_readout.0.weight'].to(device),
                           weights['deconv_readout.0.bias'].to(device))
            out = F.conv2d(out, weights['deconv_readout.1.weight'].to(device),
                           weights['deconv_readout.1.bias'].to(device), padding=(1, 1))
            out = F.sigmoid(out)

            # out = out.squeeze(1)

        return out 

    def network_forward(self, x, weights=None):
        return self.forward(x, weights)

    def copy_weights(self, network):
        for m_from, m_to in zip(network.modules(), self.modules()):
            if isinstance(m_to, nn.Conv2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
            if isinstance(m_to, nn.ConvTranspose2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


# model = DenseEDNet()
# model_dict = model.state_dict()
# pretrained_dict = torch.load('./dense_weights/dense161_features.pth')
#
#
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#
# model_dict.update(pretrained_dict)
#
# model.load_state_dict(model_dict)
# test = np.zeros((1, 3, 256, 256))
# print(model(torch.FloatTensor(test)).size())
# print(model.state_dict().keys())

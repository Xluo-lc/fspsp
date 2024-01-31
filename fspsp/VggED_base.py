import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VggEDBase(nn.Module):

    def __init__(self):
        super(VggEDBase, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1))  # 64 * 256 * 256
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 256* 256
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=2)  # pooling 64 * 128 * 128

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1))  # 128 * 128 * 128
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 128 * 128
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=2)  # pooling 128 * 64 * 64

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1))  # 256 * 64 * 64
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 64 * 64
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 64 * 64
        self.maxpool3 = nn.MaxPool2d((2, 2), stride=2)  # pooling 256 * 32 * 32

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=(1, 1))  # 512 * 32 * 32
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 32 * 32
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 32 * 32
        self.maxpool4 = nn.MaxPool2d((2, 2), stride=2)  # pooling 512 * 16 * 16

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 16 * 16
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 16 * 16
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 16 * 16

        for param in self.parameters():
            param.requires_grad = False
        # for param in self.parameters():
        #     param.requires_grad = False
        # Decoding Block


        self.deconv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.deconv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.deconv5 = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=(1, 1)),
            nn.Sigmoid()
        )

        # self.deconv = nn.Sequential(
        #     nn.Conv2d(512, 256, 3, padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(256, 128, 3, padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(128, 64, 3, padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(64, 32, 3, padding=(1, 1)),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(32, 1, 3, padding=(1, 1)),
        #     nn.Sigmoid()
        # )

    def forward(self, x, weights=None):
        out = self.conv1_1(x)  # 256
        out = F.relu(out, inplace=True)
        out = self.conv1_2(out)  # 256
        out = F.relu(out, inplace=True)
        out = self.maxpool1(out)  # 128

        out = self.conv2_1(out)  # 128
        out = F.relu(out, inplace=True)
        out = self.conv2_2(out)  # 128
        out = F.relu(out, inplace=True)
        conv2 = self.maxpool2(out)  # 64

        out = self.conv3_1(conv2)  # 64
        out = F.relu(out, inplace=True)
        out = self.conv3_2(out)  # 64
        out = F.relu(out, inplace=True)
        out = self.conv3_3(out)  # 64
        out = F.relu(out, inplace=True)
        conv3 = self.maxpool3(out)  # 32

        out = self.conv4_1(conv3)  # 32
        out = F.relu(out, inplace=True)
        out = self.conv4_2(out)  # 32
        out = F.relu(out, inplace=True)
        out = self.conv4_3(out)  # 32
        out = F.relu(out, inplace=True)
        conv4 = self.maxpool4(out)  # 32

        out = self.conv5_1(conv4)  # 32
        out = F.relu(out, inplace=True)
        out = self.conv5_2(out)  # 32
        out = F.relu(out, inplace=True)
        out = self.conv5_3(out)  # 32

        # out = self.deconv(out)

        if weights is None:
            out = self.deconv1(out)
            out = self.deconv2(out)
            out = self.deconv3(out)
            out = self.deconv4(out)
            out = self.deconv5(out)
        else:
            out = F.conv2d(out, weights['deconv1.0.weight'].to(device),
                           weights['deconv1.0.bias'].to(device), padding=(1, 1))
            out = F.relu(out, inplace=True)
            out = F.upsample(out, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            out = F.conv2d(out, weights['deconv2.0.weight'].to(device),
                           weights['deconv2.0.bias'].to(device), padding=(1, 1))
            out = F.relu(out, inplace=True)
            out = F.upsample(out, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            out = F.conv2d(out, weights['deconv3.0.weight'].to(device),
                           weights['deconv3.0.bias'].to(device), padding=(1, 1))
            out = F.relu(out, inplace=True)
            out = F.upsample(out, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            out = F.conv2d(out, weights['deconv4.0.weight'].to(device),
                           weights['deconv4.0.bias'].to(device), padding=(1, 1))
            out = F.relu(out, inplace=True)
            out = F.upsample(out, size=None, scale_factor=2, mode='bilinear', align_corners=None)

            out = F.conv2d(out, weights['deconv5.0.weight'].to(device),
                           weights['deconv5.0.bias'].to(device), padding=(1, 1))
            out = F.sigmoid(out)

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


# model = VggEDBase()
# model_dict = model.state_dict()
# pretrained_dict = torch.load('/home/wzq/PycharmProjects/new-master/vgg16_conv.pth')
# #
# #
# # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# #
# model_dict.update(pretrained_dict)
# #
# model.load_state_dict(model_dict)
# # test = np.zeros((1, 3, 256, 256))
# # print(model(torch.FloatTensor(test)).size())
# print(model.state_dict().keys())

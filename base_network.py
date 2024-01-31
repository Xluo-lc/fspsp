from DenseEDModel import DenseEDNet
from dataloader import Mydataset
from torch.utils import data
import torch
from utils import *
from VggED_base import VggEDBase
from collections import OrderedDict
import logging
import os
from model import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaseNetwork(VGGModel):
    def __init__(self, lr_alpha, meta_batch):
        super(BaseNetwork, self).__init__()

        # self.loss_function = loss_function
        self.lr_alpha = lr_alpha
        self.meta_batch = meta_batch

        # for param in self.dense.parameters():
        #     param.requires_grad = False

    def network_forward(self, x, weights=None):
        return super(BaseNetwork, self).forward(x, weights)

    # 用于这个类的forward函数中的前馈计算
    def forward_pass(self, img, sal_map, weights=None):
        img = img.to(device)
        sal_map = sal_map.to(device)

        out = self.network_forward(img, weights)
        loss = loss_function(out, sal_map)
        #print(loss)
        return loss, out

    def forward(self, sal_maps_tra, sal_maps_val):
        # print('base: sal_maps_tra', sal_maps_tra)
        data_tra = Mydataset(sal_maps_tra,
                             images_path=r'C:\Users\18817\Documents\PHcode\fspsp\PSM\all_images_release')
        dataloader_tra = data.DataLoader(data_tra, batch_size=1, shuffle=True)
        # print('base: sal_maps_val', sal_maps_val)
        data_val = Mydataset(sal_maps_val,
                             images_path=r'C:\Users\18817\Documents\PHcode\fspsp\PSM\all_images_release')
        dataloader_val = data.DataLoader(data_val, batch_size=1, shuffle=True)
        # self.eval()     1)backbone中含有batchnorm，所以觉得前馈需要设置eval模式
        loss_tra_pre, KLD_tra_pre, CC_tra_pre, SIM_tra_pre = evaluate(self, dataloader_tra)  # 前馈前的测试，不需要weights（构建functional层的wights=None），直接是导入已有权重算起来
        loss_val_pre, KLD_val_pre, CC_val_pre, SIM_val_pre = evaluate(self, dataloader_val)

        base_weights = OrderedDict(
            (name, parameter) for (name, parameter) in self.named_parameters() if parameter.requires_grad
        )#OrderedDict，实现了对字典对象中元素的排序
        #print(base_weights.keys())
        # base_weights = OrderedDict(
        #     (name, parameter) for (name, parameter) in self.named_parameters()
        # )

        # self.train()
        for index, (img, sal_map) in enumerate(dataloader_tra):
            img = img.to(device)
            sal_map = sal_map.to(device)

            if index == 0:
                trainable_weights = [p for n, p in self.named_parameters() if p.requires_grad]
                # trainable_weights = [p for n, p in self.named_parameters()]
                loss, _ = self.forward_pass(img, sal_map)
                # print(loss)
                gradients = torch.autograd.grad(loss, trainable_weights, create_graph=True)
                # print(gradients)
            else:
                trainable_weights = [v for k, v in base_weights.items() if 'deconv' in k]
                # trainable_weights = [v for k, v in base_weights.items()]
                loss, _ = self.forward_pass(img, sal_map, base_weights)
                # print(loss)
                gradients = torch.autograd.grad(loss, trainable_weights, create_graph=True)
                # print('222222222', gradients)

            # trainable_weights = [p for n, p in self.named_parameters() if p.requires_grad]
            # # trainable_weights = [p for n, p in self.named_parameters()]
            # loss, _ = self.forward_pass(img, sal_map)
            # print(loss)
            # gradients = torch.autograd.grad(loss, trainable_weights, create_graph=True)

            base_weights = OrderedDict((name, parameter - self.lr_alpha * gradient) for ((name, parameter), gradient) in
                                       zip(base_weights.items(), gradients))

        # print(base_weights)
        # print(self.named_parameters())
        # img, sal_map = dataloader_tra.__iter__().next()
        # trainable_weights = [v for k, v in self.named_parameters() if v.requires_grad]
        # loss, _ = self.forward_pass(img, sal_map, base_weights)
        # print(loss)
        # gradients = torch.autograd.grad(loss, trainable_weights, create_graph=True)
        # print(gradients)

        # loss, _ = self.forward_pass(img, sal_map, base_weights)
        # print(loss)
        #
        # trainable_weights = {n: p for n, p in self.named_parameters() if p.requires_grad}
        # gradients = torch.autograd.grad(loss, trainable_weights.values())
        # print(gradients)

        loss_tra_post, KLD_tra_post, CC_tra_post, SIM_tra_post = evaluate(self, dataloader_tra, weights=base_weights)
        loss_val_post, KLD_val_post, CC_val_post, SIM_val_post = evaluate(self, dataloader_val, weights=base_weights)

        logging.info("==========================")
        logging.info("(Meta-training) tra_pre loss: {}, KLD: {}, CC: {}, SIM: {}".format(loss_tra_pre, KLD_tra_pre,
                                                                                           CC_tra_pre, SIM_tra_pre))
        logging.info(
            "(Meta-training) tra_post loss: {}, KLD: {}, CC: {}, SIM: {}".format(loss_tra_post, KLD_tra_post,
                                                                                    CC_tra_post, SIM_tra_post))
        logging.info(
            "(Meta-training) val_pre loss: {}, KLD: {}, CC: {}, SIM: {}".format(loss_val_pre, KLD_val_pre,
                                                                                CC_val_pre, SIM_val_pre))
        logging.info(
            "(Meta-training) val_post loss: {}, KLD: {}, CC: {}, SIM: {}".format(loss_val_post, KLD_val_post,
                                                                                CC_val_post, SIM_val_post))
        logging.info("==========================")

        img, sal_map = dataloader_val.__iter__().next()
        #sal_map = sal_map.float().unsqueeze(0).to(device)
        #print('-------------------------------sal_maps:', sal_map.size())
        loss, _ = self.forward_pass(img, sal_map, base_weights)
        # print(loss)
        loss /= self.meta_batch
        #print(loss)

        trainable_weights = {n: p for n, p in self.named_parameters() if p.requires_grad}
        # trainable_weights = {n: p for n, p in base_weights.items() if 'dense' not in n}
        gradients = torch.autograd.grad(loss, trainable_weights.values())
        # print(gradients)
        meta_gradients = {name: grad for ((name, _), grad) in zip(trainable_weights.items(), gradients)}

        metrics = (loss_tra_post, KLD_tra_post, CC_tra_post, SIM_tra_post, loss_val_post, KLD_val_post, CC_val_post, SIM_val_post)
        return metrics, meta_gradients, loss



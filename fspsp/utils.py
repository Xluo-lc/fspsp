import torch
import torch.nn as nn
from loss import *
import os
import tqdm
import cv2, os
import torch
from os.path import join
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def binary_cross_entropy(x, y):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()

def loss_function(sp, sal_map):
    #loss = kldiv(sp, sal_map) - cc(sp, sal_map)
    #bce = nn.MSELoss()
    #bce = binary_cross_entropy(sp ,sal_map)
    #loss = bce(sp, sal_map) - cc(sp, sal_map)

    mse = nn.MSELoss()
    kld = nn.KLDivLoss()
    # bce = nn.BCELoss()
    loss = mse(sp, sal_map) + kld(sp.squeeze(1).log(), sal_map.squeeze(1)) - cc(sp, sal_map) - similarity(sp, sal_map)

    return loss


# 用在下面的evaluate函数和DenseEDModel的实例
def forward_pass(network, img, sal_map, weights=None):
    img = img.to(device)
    sal_map = sal_map.to(device)

    out = network.network_forward(img, weights)
    # print('out:', out.size())
    # print('sal_map:', sal_map.size())
    loss = loss_function(out, sal_map)

    return out, loss


def evaluate(network, dataloader, weights=None):
    loss, KLD, CC, SIM = 0.0, 0.0, 0.0, 0.0
    for _, (img, sal_map) in enumerate(dataloader):
        out, _ = forward_pass(network, img, sal_map, weights)
        sal_map = sal_map.to(device)
        # print(out.device)
        # print(sal_map.device)
        loss = loss_function(out, sal_map)
        KLD = kldiv(out, sal_map)
        CC = cc(out, sal_map)
        SIM = similarity(out, sal_map)

        loss += loss.item()
        KLD += KLD.item()
        CC += CC.item()
        SIM += SIM.item()

    loss /= len(dataloader)
    KLD /= len(dataloader)
    CC /= len(dataloader)
    SIM /= len(dataloader)

    return loss, KLD, CC, SIM


class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)


def visualize_model(model, loader, device, args):
    with torch.no_grad():
        model.eval()
        os.makedirs(args.results_dir, exist_ok=True)

        for (img, img_id, sz) in tqdm(loader):
            img = img.to(device)

            pred_map = model(img)
            pred_map = pred_map.cpu().squeeze(0).numpy()
            pred_map = cv2.resize(pred_map, (sz[0], sz[1]))

            pred_map = torch.FloatTensor(blur(pred_map))
            img_save(pred_map, join(args.results_dir, img_id[0]), normalize=True)


def img_save(tensor, fp, nrow=8, padding=2,
             normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                           normalize=normalize, range=range, scale_each=scale_each)

    ''' Add 0.5 after unnormalizing to [0, 255] to round to nearest integer '''

    ndarr = torch.round(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    exten = fp.split('.')[-1]
    if exten == "png":
        im.save(fp, format=format, compress_level=0)
    else:
        im.save(fp, format=format, quality=100)  # for jpg
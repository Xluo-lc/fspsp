import argparse
import glob, os
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal as Norm
from torch.autograd import Variable
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np, cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import multivariate_normal
#from dataloader import TestLoader, SaliconDataset
from loss import *
from tqdm import tqdm
from utils import *
from model import *
from dataloader import *
from VggED_base import *

parser = argparse.ArgumentParser()

parser.add_argument('--val_img_dir', default="C:\\Users\\18817\\Documents\\PHcode\\DatasetAll\\SALICON\\images\\val\\", type=str)
parser.add_argument('--model_val_path', default="./weights/epoch_496.pt", type=str)
parser.add_argument('--no_workers', default=0, type=int)
parser.add_argument('--enc_model', default="pnas", type=str)
parser.add_argument('--results_dir', default="./results/", type=str)
parser.add_argument('--validate', default=1, type=int)
parser.add_argument('--save_results', default=0, type=int)
parser.add_argument('--dataset_dir', default='C:\\Users\\18817\\Documents\\PHcode\\DatasetAll\\SALICON\\', type=str)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = VggEDBase()

# model.load_state_dict(torch.load(args.model_val_path), False)
model.load_state_dict(torch.load(args.model_val_path))
model = model.to(device)

val_img_ids = os.listdir(args.val_img_dir)
val_dataset = TestLoader(args.val_img_dir, val_img_ids)
vis_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)


def validate(model, loader, device, args):
    print("start")
    model.eval()
    tic = time.time()
    total_loss = 0.0
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()

    for (img, gt) in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)
        # fixations = fixations.to(device)

        pred_map = model(img)

        # Blurring
        blur_map = pred_map.cpu().squeeze(0).clone().numpy()
        blur_map = blur(blur_map).unsqueeze(0).to(device)

        cc_loss.update(cc(blur_map, gt))
        kldiv_loss.update(kldiv(blur_map, gt))
        nss_loss.update(nss(blur_map, gt))
        sim_loss.update(similarity(blur_map, gt))

    print('CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(cc_loss.avg,
                                                                                               kldiv_loss.avg,
                                                                                               nss_loss.avg,
                                                                                               sim_loss.avg, (
                                                                                                           time.time() - tic) / 60))
    sys.stdout.flush()

    return cc_loss.avg


if args.validate:
    val_img_dir = args.dataset_dir + "images\\val\\"
    val_gt_dir = args.dataset_dir + "maps\\val\\"
    val_fix_dir = args.dataset_dir + "fixations\\fixations\\"

    val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
    val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)
    with torch.no_grad():
        validate(model, val_loader, device, args)
if args.save_results:
    visualize_model(model, vis_loader, device, args)

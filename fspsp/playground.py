from dataloader import Mydataset
from torch.utils import data
import random
import os
import math

# sal_maps_tra = ['/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_1/53106.png', '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_1/00380.png', '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_1/idx_317_n07742313_8087.png', '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_1/captainamerica-civilwarofficialtrailer12016-chrisevansscarlettjohanssonmoviehd.mp4-00.01.22.791.png', '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_1/newspectretrailer.mp4-00.02.15.677.png']
# sal_maps_val = ['/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_5/atonementofficialtrailer1-brendablethynmovie2007hd.mp4-00.00.15.708.png', '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_5/n02802426_3683.png', '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_5/idx_179_ilsvrc2014_train_00030330.png', '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_5/idx_12_ilsvrc2014_train_00001670.png', '/home/wzq/Second_project/PSM_dataset/fixation_map_30_release/train/Sub_5/dscf1486.png']
# data_tra = Mydataset(sal_maps_tra,
#                      images_path='/home/wzq/Second_project/PSM_dataset/all_images_release')
# dataloader_tra = data.DataLoader(data_tra, batch_size=1, shuffle=True)
# data_val = Mydataset(sal_maps_val,
#                              images_path='/home/wzq/Second_project/PSM_dataset/all_images_release')
# dataloader_val = data.DataLoader(data_val, batch_size=1, shuffle=True)



# x = [1, 5, 4]
# y = [1, 2, 3, 4, 5, 6]
# print([y[i] for i in x])
# import torch
# import torch.nn.functional as F
#
# x = torch.tensor([[[1., 2., 3., 4.], [1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]])
# print(x.size())
# out = F.max_pool2d(x, kernel_size=2)
# print(out)

from VggED_base import VggEDBase
from collections import OrderedDict #对字典对象中元素的排序
model = VggEDBase()
base_weights = OrderedDict(
            (name, parameter) for (name, parameter) in model.named_parameters() if parameter.requires_grad
        )
print(base_weights.keys())
trainable_weights = {k: v for k, v in base_weights.items() if 'conv1' not in k}
print(trainable_weights.keys())


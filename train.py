import random
import math
from torch.utils import data
from dataloader import Mydataset
from Learner import MetaLearner
import logging
import time
import time
from utils import loss_function
from tqdm import tqdm
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


# EPOCH = 1
# # task_tra = 25  # number of the tasks for meta-train
# # task_test = 5  # number of the tasks for meta-test
# task_batch = 1  # number of the tasks in each batch
# shot = 5  # number of samples in each task

# def get_task_seq(salmap_path):
#     # img_task_path = [os.path.join(img_path, task) for task in os.listdir(img_path)]
#     task_path = [os.path.join(salmap_path, task) for task in os.listdir(salmap_path)]
#     # fixation_path =
#     random.shuffle(task_path)
#
#     # for mac
#     for i in range(len(task_path)):
#         if task_path[i].split('/')[-1] == '.DS_Store':
#             task_path.remove(task_path[i])
#             break
#
#     return task_path
#
# count = 0
# for epoch in range(EPOCH):
#     task_seq = get_task_seq('/Users/wangziqiang/Desktop/SaliencyPrediction/dataset/personalized/fixation_map_30_release')
#     print('task_seq: ', len(task_seq))
#     for batch in range(math.ceil(len(task_seq) / task_batch)):
#         if batch == len(task_seq) // task_batch:
#             tasks = task_seq[batch * task_batch: len(task_seq)]
#         else:
#             tasks = task_seq[batch * task_batch: (batch + 1) * task_batch]
#
#         sal_maps_tra = []
#         sal_maps_val = []
#         for task in tasks:
#             sal_maps = [os.path.join(task, sal_map) for sal_map in os.listdir(task)]
#             random.shuffle(sal_maps)
#             sal_maps_tra += sal_maps[0: shot]
#             sal_maps_val += sal_maps[shot: shot * 2]
#
#         data_tra = Mydataset(sal_maps_tra, images_path='/Users/wangziqiang/Desktop/SaliencyPrediction/dataset/personalized/all_images_release/all_images_release')
#         dataloader_tra = data.DataLoader(data_tra, batch_size=1, shuffle=True)
#         # for _, (image, sal_map) in enumerate(dataloader_tra):
#         for index in range(6):
#             image, sal_map = dataloader_tra.__iter__().__next__()
#             print('img: ', type(image))
#
#             print(sal_map.size())
#             # cv2.imshow('map', sal_map)
#             # cv2.waitKey(0)
#             # print('sal: ', sal_map)
#             count += 1
#
# print(count)

logging.basicConfig(filename=r'C:\\Users\\18817\\Documents\\PHcode\\second-master\\log\\{}'.format(time.strftime('%Y-%m-%d-%H_%M_%S',
                                                                        time.localtime())), level=logging.INFO)
logging.info('Started training')
meta_learner = MetaLearner(meta_updates=501, meta_batch=5, second_meta_batch=2, lr_alpha=0.00001, lr_beta=0.00001,
                           save_path='./vggweights_11_19_5way1shot45/', shot=1)

meta_learner.train_maxcc50()
# meta_learner.test_maxcc50()
logging.info('Finished training')

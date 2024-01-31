from torch.utils import data

from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import torch
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Mydataset(data.Dataset):
    def __init__(self, sal_maps, images_path):
        super(Mydataset, self).__init__()
        self.images_path = images_path
        self.sal_maps = sal_maps

    def __len__(self):
        return len(self.sal_maps)

    def __getitem__(self, index):
        sal_map = Image.open(self.sal_maps[index]).convert('L')
        name = self.sal_maps[index].split('\\')[-1]
        image_path = os.path.join(self.images_path, self.sal_maps[index].split('\\')[-1].replace('.png', '.jpg'))#split('\\')注意
        image = Image.open(image_path).convert('RGB')

        trans = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        image = trans(image)

        sal_map = np.array(sal_map)
        sal_map = sal_map.astype('float')
        sal_map = cv2.resize(sal_map, (256, 256))
        sal_map = sal_map / 255.0
        sal_map = torch.FloatTensor(sal_map).unsqueeze(0)  # model输出三维(1, w, h)

        return image, sal_map

def load_data(img_path):
    map_path = img_path.replace('images', 'maps').replace('.jpg', '.png')
    img = Image.open(img_path).convert('RGB')
    map = Image.open(map_path).convert('L')
    map = np.array(map)
    map = map.astype('float')
    map = cv2.resize(map, (63, 63))
    map = map / 255.0
    map = torch.FloatTensor(map).unsqueeze(0)
    return img, map

class SalicondDataset(data.Dataset):
    def __init__(self,images_path, mode, transform = None ):
        self.images_path = images_path
        #self.map_path = map_path
        self.images = self.load_images()
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def load_images(self):
        images = [os.path.join(self.images_path, img) for img in os.listdir(self.images_path)]
        return images


    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.images[index]
            img, map = load_data(img_path)
            image = self.transform(img)
            return image, map


class TestLoader(data.DataLoader):
    def __init__(self, img_dir, img_ids):
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert('RGB')
        sz = img.size
        img = self.img_transform(img)
        return img, img_id, sz

    def __len__(self):
        return len(self.img_ids)


class SaliconDataset(data.DataLoader):
    def __init__(self, img_dir, gt_dir, fix_dir, img_ids, exten='.png'):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.fix_dir = fix_dir
        self.img_ids = img_ids
        self.exten = exten
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        gt_path = os.path.join(self.gt_dir, img_id + self.exten)
        fix_path = os.path.join(self.fix_dir, img_id + self.exten)

        img = Image.open(img_path).convert('RGB')

        gt = np.array(Image.open(gt_path).convert('L'))
        gt = gt.astype('float')
        gt = cv2.resize(gt, (256, 256))

        # fixations = np.array(Image.open(fix_path).convert('L'))
        # fixations = fixations.astype('float')

        img = self.img_transform(img)
        if np.max(gt) > 1.0:
            gt = gt / 255.0
        # fixations = (fixations > 0.5).astype('float')

        assert np.min(gt) >= 0.0 and np.max(gt) <= 1.0  # 在表达式条件为 false 的时候触发异常
        # assert np.min(fixations)==0.0 and np.max(fixations)==1.0
        return img, torch.FloatTensor(gt)  # , torch.FloatTensor(fixations)

    def __len__(self):
        return len(self.img_ids)


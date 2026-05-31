r""" Chest X-ray few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

import albumentations as A
import random


class DatasetLungMPA(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, num=1):
        self.split = split
        self.benchmark = 'lung'
        self.shot = shot
        self.num = num

        self.base_path = os.path.join(datapath, 'LungSegmentation')
        self.img_path = os.path.join(self.base_path, 'CXR_png')
        self.ann_path = os.path.join(self.base_path, 'masks')

        self.categories = ['1']

        self.class_ids = range(0, 1)
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.transform = transform

        self.q_colorAug = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=30,val_shift_limit=20,always_apply=False,p=1)
        ])

    def __len__(self):
        return self.num
    
    def create_unique_transforms(self, n):
        transform_list = []
        seen_combinations = set()
        
        while len(transform_list) < n:
            if len(transform_list) == 0:
                transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5)
                ])
            elif len(transform_list) == 1:
                transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(),
                ])
            else:
                horizontal_flip_prob = random.choice([0.5, 0.8])
                vertical_flip_prob = random.choice([0.5, 0.8])
                shift_limit = random.choice([0.05, 0.1])
                scale_limit = random.choice([0.1, 0.2])
                rotate_limit = random.choice([10, 15, 20])
                grid_distort_limit = random.choice([0.3, 0.5])

                transform = A.Compose([
                    A.HorizontalFlip(p=horizontal_flip_prob),
                    A.VerticalFlip(p=vertical_flip_prob),
                    A.RandomRotate90(),
                    A.ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, p=0.7),
                    A.GridDistortion(num_steps=5, distort_limit=grid_distort_limit, p=0.7),
                ])

            if len(transform_list) > 1:
                transform_key = (horizontal_flip_prob, vertical_flip_prob, shift_limit, scale_limit, rotate_limit, grid_distort_limit)
                if transform_key not in seen_combinations:
                    seen_combinations.add(transform_key)
                    transform_list.append(transform)
            else:
                transform_list.append(transform)

        return transform_list

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        _, _, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_num = 6
        q_transform_list = self.create_unique_transforms(query_num)
        query_aug_imgs = []
        query_aug_masks = []

        for q_aug_transform in q_transform_list:
            if self.shot == 1:
                idx = 0
            else:
                idx = random.randint(0, self.shot-1)

            q_img = support_imgs[idx]
            q_img = np.array(q_img)
            q_mask = support_masks[idx]
            q_mask = np.array(Image.open(q_mask).convert('L'))
            pair_transform = q_aug_transform(image=q_img, mask=q_mask)
            query_img = pair_transform['image']
            query_mask = pair_transform['mask']
            q_img_transform = self.q_colorAug(image=query_img)
            query_img = q_img_transform['image']
            query_img = Image.fromarray(query_img)
            query_img = self.transform(query_img)
            query_mask = self.process_mask(query_mask)
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
            query_mask = query_mask.long()

            query_aug_imgs.append(query_img)
            query_aug_masks.append(query_mask)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        query_imgs = torch.stack(query_aug_imgs)
        query_masks = torch.stack(query_aug_masks)

        support_masks_tmp = []
        for smask in support_masks:
            smask = self.read_mask(smask)
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        return support_imgs, support_masks, query_imgs, query_masks, class_sample, support_names, query_name

    def load_frame(self, query_name, support_names):
        query_mask = self.read_mask(query_name)
        support_masks = support_names

        query_id = query_name[:-9] + '.png'
        query_img = Image.open(os.path.join(self.img_path, os.path.basename(query_id))).convert('RGB')

        support_ids = [os.path.basename(name)[:-9] + '.png' for name in support_names]
        support_names = [os.path.join(self.img_path, sid) for sid in support_ids]
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def process_mask(self, img):
        mask = torch.tensor(img)
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id

    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            os.path.join(self.base_path, cat)
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.img_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata.append(img_path)
        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % self.ann_path)])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata_classwise[cat] += [img_path]
        return img_metadata_classwise

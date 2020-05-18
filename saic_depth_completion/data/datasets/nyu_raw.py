import os
import torch

import numpy as np
from PIL import Image
import random

ROOT = '/Vol1/dbstore/datasets/depth_completion/NYUv2_raw'

class NYURaw():
    def __init__(self, split, dt=0.01, valid_split=0.05,
                 focal=None, image_aug=None,
                 depth_aug=None, geometry_aug=None,
                 n_scenes=None):

        super().__init__()

        self.split = split

        self.fx = 5.1885790117450188e+02
        self.fy = 5.1946961112127485e+02
        self.cx = 3.2558244941119034e+02
        self.cy = 2.5373616633400465e+02

        self.crop = 8

        self.focal = focal

        self.image_aug = image_aug
        self.depth_aug = depth_aug
        self.geometry_aug = geometry_aug

        self.train_path = os.path.join(ROOT, "train")
        self.test_path = os.path.join(ROOT, "test")

        with open(os.path.join(self.train_path, 'time_diffs.pth'), 'rb') as fin:
            train_time_diffs = torch.load(fin)

        with open(os.path.join(self.test_path, 'time_diffs.pth'), 'rb') as fin:
            test_time_diffs = torch.load(fin)

        train_keys = []
        for key in train_time_diffs:
            time_lapse = train_time_diffs[key]
            if abs(time_lapse) < dt:
                train_keys.append(key)

        test_keys = []
        for key in test_time_diffs:
            time_lapse = test_time_diffs[key]
            if abs(time_lapse) < dt:
                test_keys.append(key)

        if n_scenes is not None:
            new_train_keys = []
            for key in train_keys:
                if int(key.split('_')[0]) < n_scenes:
                    new_train_keys.append(key)
            train_keys = new_train_keys

        random.seed(0)
        random.shuffle(train_keys)
        n_valid = int(valid_split * len(train_keys))

        valid_keys = train_keys[:n_valid]
        train_keys = train_keys[n_valid:]

        random.shuffle(train_keys)
        random.shuffle(valid_keys)
        random.shuffle(test_keys)

        self.keys = {
            'train': train_keys,
            'val': valid_keys,
            'test': test_keys}

    def __len__(self):
        return len(self.keys[self.split])

    def __getitem__(self, index):
        key = self.keys[self.split][index]

        if self.split == 'test':
            path = self.test_path
        else:
            path = self.train_path

        image_path = os.path.join(path, 'images', key)
        depth_path = os.path.join(path, 'depths', key)

        image = torch.load(image_path)['image']
        depth = torch.load(depth_path)['depth'].astype('float32') / (2.0 ** 16 - 1.0) * 10.0

        image = image[self.crop:-self.crop, self.crop:-self.crop]
        depth = depth[self.crop:-self.crop, self.crop:-self.crop]

        depth = np.expand_dims(depth, -1)

        if self.image_aug is not None:
            if self.split in self.image_aug:
                image = self.image_aug[self.split](image=image)['image']


        # borders are invalid pixels by default
        mask = (depth > 1.0e-4) & (depth < 10.0 - 1.0e-4)

        mask[:50, :]  = 0
        mask[:, :40]  = 0
        mask[-10:, :] = 0
        mask[:, -40:] = 0

        mask = mask.astype('float32')
        center = torch.tensor([self.cx, self.cy])
        focal = torch.tensor([self.fx, self.fy])

        if self.focal is not None:
            scale = self.focal / self.fx

            interpolation = lambda x: torch.nn.functional.interpolate(
                torch.tensor(x).permute(2, 0, 1).unsqueeze(0),
                scale_factor=scale,
                mode='bilinear',
                align_corners=True)[0].permute(1, 2, 0).numpy()

            image = interpolation(image)
            depth = interpolation(depth)
            mask = interpolation(mask)
            center = center * scale
            focal = focal * scale



        if self.geometry_aug is not None:
            if self.split in self.geometry_aug:
                res = self.geometry_aug[self.split](image=image, depth=depth, mask=mask, keypoints=[center])
                image = res['image']
                depth = res['depth']
                mask = res['mask']
                center = res['keypoints'][0]

        if self.depth_aug is not None:
            if self.split in self.depth_aug:
                res = self.depth_aug[self.split](image=depth, mask=mask)
                depth = res['image']
                mask = res['mask']

        mask = (mask > 1.0 - 1.0e-4)

        sample = {
            'image': torch.tensor(image).permute(2, 0, 1).float(),
            'depth': torch.tensor(depth).permute(2, 0, 1).float(),
            'mask': torch.tensor(mask).permute(2, 0, 1).bool(),
#             'type': torch.tensor(0), # 0 stands for absolute depth
#             'focal': focal.float(),
#             'center': center.float()
        }

        return sample

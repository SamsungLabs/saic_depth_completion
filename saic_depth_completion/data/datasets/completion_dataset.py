import numpy as np
from skimage.filters import gaussian
import torch


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def create_holes_mask(layer, granularity, percentile):
    gaussian_layer = np.random.uniform(size=layer.shape[1:])
    gaussian_layer = gaussian(gaussian_layer, sigma=granularity)
    threshold = np.percentile(gaussian_layer.reshape([-1]), 100 * (1 - percentile))
    gaussian_layer = torch.tensor(gaussian_layer).unsqueeze(0)
    return gaussian_layer > threshold


def spatter(layer, mask, granularity=10, percentile=0.4):
    holes_mask = create_holes_mask(layer, granularity, percentile)

    res = layer.clone()
    mask = mask.clone()
    res[holes_mask] = 0
    mask[holes_mask] = 0
    return res, mask


def deform(layer, mask, granularity=10, percentile=0.02):
    holes_mask = create_holes_mask(layer, granularity, percentile)

    res = layer.clone()
    mask = mask.clone()
    v = res[(res > 1.0e-4) & holes_mask].mean() * 2 ** (np.random.uniform() * 2.0 - 1.0)
    res[(res > 1.0e-4) & holes_mask] = v
    mask[(res > 1.0e-4) & holes_mask] = 0

    return res, mask


class CompletionDataset:
    def __init__(self,
                 ds,
                 threshold=True,
                 granularity=8,
                 percentile_void=0.3,
                 percentile_deform=0.02):
        self.ds = ds
        self.threshold = threshold
        self.granularity = granularity
        self.percentile_deform = percentile_deform
        self.percentile_void = percentile_void

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        res = self.ds[index]
        np.random.seed(index)

        if 'raw_depth' not in res:
            res['raw_depth'] = res['depth'].clone()
            res['raw_depth_mask'] = res['mask'].clone()

        if self.threshold:
            maxd = res['raw_depth'][res['raw_depth_mask']].max()
            mind = res['raw_depth'][res['raw_depth_mask']].min()
            threshold = np.random.uniform() * (maxd - mind) + mind
            mask = (res['raw_depth'] > threshold)
            res['raw_depth'][mask] = 0

        res['raw_depth'], res['raw_depth_mask'] = deform(res['raw_depth'],
                                                         res['raw_depth_mask'],
                                                         granularity=self.granularity,
                                                         percentile=self.percentile_deform)
        res['raw_depth'], res['raw_depth_mask'] = spatter(res['raw_depth'],
                                                          res['raw_depth_mask'],
                                                          granularity=self.granularity,
                                                          percentile=self.percentile_void)
        res['gt_depth'] = res.pop('depth')
        res['color'] = res.pop('image')
        return res
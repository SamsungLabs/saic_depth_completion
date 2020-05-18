import numpy as np
import matplotlib.pyplot as plt

def figure(color, raw_depth, mask, gt, pred, close=False):
    fig, axes = plt.subplots(3, 2, figsize=(7, 10))

    color       = color.cpu().permute(1, 2, 0)
    raw_depth   = raw_depth.cpu()
    mask        = mask.cpu()
    gt          = gt.cpu()
    pred        = pred.detach().cpu()

    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())


    axes[0, 0].set_title('RGB')
    axes[0, 0].imshow((color - color.min()) / (color.max() - color.min()) )

    axes[0, 1].set_title('raw_depth')
    img = axes[0, 1].imshow(raw_depth[0], cmap='RdBu_r')
    fig.colorbar(img, ax=axes[0, 1])

    axes[1, 0].set_title('mask')
    axes[1, 0].imshow(mask[0])

    axes[1, 1].set_title('gt')
    img = axes[1, 1].imshow(gt[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=axes[1, 1])

    axes[2, 1].set_title('pred')
    img = axes[2, 1].imshow(pred[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=axes[2, 1])
    if close: plt.close(fig)
    return fig
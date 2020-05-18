import torch.utils.tensorboard as tb

from saic_depth_completion.utils import visualize

class Tensorboard:
    def __init__(self, tb_dir, max_figures=10):
        self.tb_dir = tb_dir
        self.max_figures = max_figures
    def update(self, metrics_dict, epoch, tag="train"):
        with tb.SummaryWriter(self.tb_dir) as writer:
            for k, v in metrics_dict.items():
                writer.add_scalar(k+"/"+tag, v, epoch)
    def add_figures(self, batch, post_pred, tag="train", epoch=0):
        with tb.SummaryWriter(self.tb_dir) as writer:
            B = batch["color"].shape[0]
            self.max_figures = min(B, self.max_figures)
            for it in range(self.max_figures):
                fig = visualize.figure(
                    batch["color"][it], batch["raw_depth"][it],
                    batch["mask"][it], batch["gt_depth"][it],
                    post_pred[it]
                )
                writer.add_figure(
                    figure=fig,
                    tag=tag + "_epoch_" + str(epoch),
                    close=True
                )

import time
import datetime
import torch
from tqdm import tqdm

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter


def validate(
        model, val_loaders, metrics, epoch=0, logger=None, tensorboard=None, tracker=None
):

    model.eval()
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    for subset, loader in val_loaders.items():
        logger.info(
            "Validate: ep: {}, subset -- {}. Total number of batches: {}.".format(epoch, subset, len(loader))
        )

        metrics_meter.reset()
        # loop over dataset
        for batch in tqdm(loader):
            batch = model.preprocess(batch)
            pred = model(batch)

            with torch.no_grad():
                post_pred = model.postprocess(pred)
                metrics_meter.update(post_pred, batch["gt_depth"])

        state = "Validate: ep: {}, subset -- {} | ".format(epoch, subset)
        logger.info(state + metrics_meter.suffix)

        metric_state = {k: v.global_avg for k, v in metrics_meter.meters.items()}

        if tensorboard is not None:
            tensorboard.update(metric_state, tag=subset, epoch=epoch)
            tensorboard.add_figures(batch, post_pred, tag=subset, epoch=epoch)

        if tracker is not None:
            tracker.update(subset, metric_state)
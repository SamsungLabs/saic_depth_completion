import time
import datetime
import torch

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from saic_depth_completion.engine.val import validate

def train(
    model, trainloader, optimizer, val_loaders={}, scheduler=None, snapshoter=None, logger=None,
    epochs=100, init_epoch=0,  logging_period=10, metrics={}, tensorboard=None, tracker=None
):

    # move model to train mode
    model.train()
    logger.info(
        "Total number of params: {}".format(model.count_parameters())
    )
    loss_meter = LossMeter(maxlen=20)
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    logger.info(
        "Start training at {} epoch. Total number of epochs {}.".format(init_epoch, epochs)
    )

    num_batches = len(trainloader)

    start_time_stamp = time.time()
    for epoch in range(init_epoch, epochs):
        loss_meter.reset()
        metrics_meter.reset()
        # loop over dataset
        for it, batch in enumerate(trainloader):
            batch = model.preprocess(batch)
            pred = model(batch)
            loss = model.criterion(pred, batch["gt_depth"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), 1)

            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                post_pred = model.postprocess(pred)
                metrics_meter.update(post_pred, batch["gt_depth"])

            if (epoch * num_batches + it) % logging_period == 0:
                state = "ep: {}, it {}/{} -- loss {:.4f}({:.4f}) | ".format(
                    epoch, it, num_batches, loss_meter.median, loss_meter.global_avg
                )
                logger.info(state + metrics_meter.suffix)

        state = "ep: {}, it {}/{} -- loss {:.4f}({:.4f}) | ".format(
            epoch, it, num_batches, loss_meter.median, loss_meter.global_avg
        )

        logger.info(state + metrics_meter.suffix)

        if tensorboard is not None:
            tensorboard.update(
                {k: v.global_avg for k, v in metrics_meter.meters.items()}, tag="train", epoch=epoch
            )
            tensorboard.add_figures(batch, post_pred, epoch=epoch)

        if snapshoter is not None and epoch % snapshoter.period == 0:
            snapshoter.save('snapshot_{}'.format(epoch))

        # validate
        # ...
        validate(
            model, val_loaders, metrics, epoch=epoch, logger=logger,
            tensorboard=tensorboard, tracker=tracker
        )

    if snapshoter is not None:
        snapshoter.save('snapshot_final')

    total_time = str(datetime.timedelta(seconds=time.time() - start_time_stamp))

    logger.info(
        "Training finished! Total spent time: {}.".format(total_time)
    )
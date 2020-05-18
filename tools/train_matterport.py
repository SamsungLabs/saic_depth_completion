import torch

import argparse

import numpy as np

from saic_depth_completion.data.datasets.matterport import Matterport
from saic_depth_completion.engine.train import train
from saic_depth_completion.utils.tensorboard import Tensorboard
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.experiment import setup_experiment
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.utils.tracker import ComposedTracker, Tracker
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.config import get_default_config
from saic_depth_completion.data.collate import default_collate
from saic_depth_completion.metrics import Miss, SSIM, DepthL2Loss, DepthL1Loss, DepthRel


def main():
    parser = argparse.ArgumentParser(description="Some training params.")
    parser.add_argument(
        "--debug", dest="debug", type=bool, default=False, help="Setup debug mode"
    )
    parser.add_argument(
        "--postfix", dest="postfix", type=str, default="", help="Postfix for experiment's name"
    )
    parser.add_argument(
        "--default_cfg", dest="default_cfg", type=str, default="arch0", help="Default config"
    )
    parser.add_argument(
        "--config_file", default="", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--snapshot_period", default=10, type=int, help="Snapshot model one time over snapshot period"
    )
    args = parser.parse_args()

    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MetaModel(cfg, device)

    logger = setup_logger()
    experiment = setup_experiment(
        cfg, args.config_file, logger=logger, training=True, debug=args.debug, postfix=args.postfix
    )


    optimizer  = torch.optim.Adam(
        params=model.parameters(), lr=cfg.train.lr
    )
    if not args.debug:
        snapshoter = Snapshoter(
            model, optimizer, period=args.snapshot_period, logger=logger, save_dir=experiment.snapshot_dir
        )
        tensorboard = Tensorboard(experiment.tensorboard_dir)
        tracker = ComposedTracker([
            Tracker(subset="test_matterport", target="mse", snapshoter=snapshoter, eps=0.01),
            Tracker(subset="val_matterport", target="mse", snapshoter=snapshoter, eps=0.01),
        ])
    else:
        snapshoter, tensorboard, tracker = None, None, None


    metrics = {
        'mse': DepthL2Loss(),
        'mae': DepthL1Loss(),
        'd105': Miss(1.05),
        'd110': Miss(1.10),
        'd125_1': Miss(1.25),
        'd125_2': Miss(1.25**2),
        'd125_3': Miss(1.25**3),
        'rel': DepthRel(),
        'ssim': SSIM(),
    }

    train_dataset = Matterport(split="train")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=default_collate
    )

    val_datasets = {
        "val_matterport": Matterport(split="val"),
        "test_matterport": Matterport(split="test"),
    }
    val_loaders = {
        k: torch.utils.data.DataLoader(
            dataset=v,
            batch_size=cfg.test.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=default_collate
        )
        for k, v in val_datasets.items()
    }

    train(
        model,
        train_loader,
        val_loaders=val_loaders,
        optimizer=optimizer,
        snapshoter=snapshoter,
        epochs=200,
        logger=logger,
        metrics=metrics,
        tensorboard=tensorboard,
        tracker=tracker
    )


if __name__ == "__main__":
    main()
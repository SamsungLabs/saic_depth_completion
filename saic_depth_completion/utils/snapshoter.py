import logging
import os
import torch

import numpy as np

class Snapshoter:
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            period=10,
            logger=None,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.period    = period

        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        if not save_dir:
            self.logger.warn(
                "Snapshoter's arg 'save_dir' was not passed. Using default value({}).".format('./')
            )
            self.save_dir = "./"
        else:
            self.save_dir = save_dir

    def save(self, fname, **kwargs):

        data = dict()
        data["model"]      = self.model.state_dict()
        data["optimizer"]  = self.optimizer.state_dict() if self.optimizer else None
        data["scheduler"]  = self.scheduler.state_dict() if self.scheduler else None

        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(fname))
        self.logger.info("Saving snapshot into {}".format(save_file))
        torch.save(data, save_file)


    def load(self, fname, model_only=False):
        if os.path.exists(fname):
            path = fname
        elif os.path.exists(os.path.join(self.save_dir, fname)):
            path = os.path.join(self.save_dir, fname)
        else:
            self.logger.info("No snapshot found. Initializing model from scratch")
            return

        self.logger.info("Loading snapshot from {}".format(path))
        snapshot = torch.load(path, map_location=torch.device("cpu"))
        self.model.load_state_dict(snapshot.pop("model"))

        if model_only: return snapshot

        if snapshot["optimizer"] is not None and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(path))
            self.optimizer.load_state_dict(snapshot.pop("optimizer"))
        if snapshot["scheduler"] is not None and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(path))
            self.scheduler.load_state_dict(snapshot.pop("scheduler"))

        return snapshot
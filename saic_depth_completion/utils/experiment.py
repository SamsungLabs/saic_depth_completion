import os
import yaml
import logging
import shutil
from easydict import EasyDict as edict
from saic_depth_completion.utils.registry import Registry


parsers = Registry()

def setup_experiment(cfg, config_file, postfix="", log_dir="./logs/", tensorboard_dir="./tensorboard/",
                     delimiter="|", logger=None, training=True, debug=False):
    if logger is None:
        logger = logging.getLogger(__name__)

    experiment = edict()
    experiment.name = delimiter.join(
        parsers[cfg.model.arch](cfg.model) + parse_train_params(cfg.train)
    )
    logger.info("Experiment name: {}".format(experiment.name))
    if postfix:
        experiment.name = experiment.name + "-" + postfix
    experiment.dir = os.path.join(
        log_dir, experiment.name
    )
    experiment.snapshot_dir = os.path.join(
        log_dir, experiment.name, "snapshots"
    )
    experiment.tensorboard_dir = os.path.join(
        tensorboard_dir, experiment.name
    )

    if not debug:
        logger.info("Experiment dir: {}".format(experiment.dir))
        os.makedirs(experiment.snapshot_dir, exist_ok=not training)
        logger.info("Snapshot dir: {}".format(experiment.snapshot_dir))
        os.makedirs(experiment.tensorboard_dir, exist_ok=not training)
        logger.info("Tensorboard dir: {}".format(experiment.tensorboard_dir))

        if training:
            shutil.copy2(config_file, experiment.dir)

    return experiment



@parsers.register("DM-LRN")
def parse_dm_lrn(model_cfg):
    model_params = [model_cfg.arch, model_cfg.modulation]
    backbone_params =  [
        model_cfg.backbone.arch,
        "imagenet" if model_cfg.backbone.imagenet else "",
        str(model_cfg.backbone.norm_layer).split('.')[-1][:-2],
    ]
    criterion = []
    for spec in model_cfg.criterion:
        if len(spec) == 3:
            criterion.append(
                "(" +
                str(spec[1]) + "*" + spec[0] +
                "(" + ",".join( [str(i) for i in spec[2]] ) +")" +
                ")"
            )
        else:
            criterion.append(
                "(" + str(spec[1]) + "*" + spec[0] + ")"
            )
    loss = "+".join(criterion)

    return model_params + backbone_params + [loss]

@parsers.register("LRN")
def parse_arch1(model_cfg):
    model_params = [
        model_cfg.arch,
        "CRP" if model_cfg.use_crp else "NoCRP"
    ]
    backbone_params =  [
        model_cfg.backbone.arch,
        "imagenet" if model_cfg.backbone.imagenet else "",
        str(model_cfg.backbone.norm_layer).split('.')[-1][:-2],
    ]
    criterion = []
    for spec in model_cfg.criterion:
        if len(spec) == 3:
            criterion.append(
                "(" +
                str(spec[1]) + "*" + spec[0] +
                "(" + ",".join( [str(i) for i in spec[2]] ) +")" +
                ")"
            )
        else:
            criterion.append(
                "(" + str(spec[1]) + "*" + spec[0] + ")"
            )
    loss = "+".join(criterion)

    return model_params + backbone_params + [loss]

def parse_train_params(train_cfg):
    train_params = [
        "lr="+str(train_cfg.lr),
    ]
    return train_params

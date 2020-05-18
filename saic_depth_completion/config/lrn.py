from yacs.config import CfgNode as CN

_C = CN()
_C.model = CN()
# global arch
_C.model.arch = 'LRN'
# width of model
_C.model.max_channels = 256
# activation: (type: str, kwargs: dict)
_C.model.activation = ("LeakyReLU", [0.2, True])
# upsample mode
_C.model.upsample = "bilinear"
# include CRP blocks or not
_C.model.use_crp = True
# loss fn: list of tuple
_C.model.criterion = [("LogDepthL1Loss", 1.0)]
_C.model.predict_log_depth = True
_C.model.input_mask = False

# backbone
_C.model.backbone = CN()
# backbone arch
_C.model.backbone.arch = 'efficientnet-b0'
# pretraining
_C.model.backbone.imagenet = True
# batch norm or frozen batch norm
_C.model.backbone.norm_layer = ""
# return features from 4 scale or not
_C.model.backbone.multi_scale_output = True

# train parameters
_C.train = CN()
# use standard scaler or not
_C.train.rgb_mean = [0.485, 0.456, 0.406]
_C.train.rgb_std = [0.229, 0.224, 0.225]
# standard scaler params for raw_depth
_C.train.depth_mean = 2.1489
_C.train.depth_std = 1.4279
_C.train.batch_size = 32
_C.train.lr = 0.0001

_C.test = CN()
_C.test.batch_size = 16

#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited.

""" Builder for video models. """
# !/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited.

""" Builder for video models. """

import sys
import torch
import torch.nn as nn

import traceback

import utils.logging as logging

from models.base.models import BaseVideoModel, MODEL_REGISTRY
from models.utils.model_ema import ModelEmaV2

logger = logging.get_logger(__name__)


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (Config): global config object that provides specifics to construct the model.
        gpu_id (Optional[int]): specify the gpu index to build model.
    Returns:
        model: constructed model
        model_ema: copied model for ema
    """
    # Construct the model
    if MODEL_REGISTRY.get(cfg.MODEL.NAME) == None:
        # attempt to find standard models
        model = BaseVideoModel(cfg)
    else:
        # if the model is explicitly defined,
        # it is directly constructed from the model pool
        model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)

    if torch.cuda.is_available():
        assert (
                cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
                cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        model = model.cuda(device=cur_device)

    model_ema = None
    if cfg.MODEL.EMA.ENABLE:
        model_ema = ModelEmaV2(model, decay=cfg.MODEL.EMA.DECAY)

    try:
        # convert batchnorm to be synchronized across
        # different GPUs if needed
        sync_bn = cfg.BN.SYNC_BN
        if sync_bn == True and cfg.NUM_GPUS * cfg.NUM_SHARDS > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    except:
        sync_bn = None

    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS * cfg.NUM_SHARDS > 1:
        # Make model replica operate on the current device
        if cfg.PAI:
            # Support distributed training on the cluster
            model = torch.nn.parallel.DistributedDataParallel(
                module=model
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device
            )

    return model, model_ema
# import sys
# import torch
# import torch.nn as nn
#
# import traceback
#
# import utils.logging as logging
#
# from models.base.models import BaseVideoModel, MODEL_REGISTRY
# from models.utils.model_ema import ModelEmaV2
#
# logger = logging.get_logger(__name__)
#
# def build_model(cfg, gpu_id=None):
#     """
#     Builds the video model.
#     Args:
#         cfg (Config): global config object that provides specifics to construct the model.
#         gpu_id (Optional[int]): specify the gpu index to build model (overrides cfg.gpu_id).
#     Returns:
#         model: constructed model
#         model_ema: copied model for EMA (Exponential Moving Average)
#     """
#
#     # Use cfg.gpu_id unless an override is given
#     gpu_ids = gpu_id if gpu_id is not None else cfg.gpu_id
#     if isinstance(gpu_ids, int):
#         gpu_ids = [gpu_ids]  # ensure list
#     assert isinstance(gpu_ids, (list, tuple)), "`gpu_id` should be an int or list of ints"
#
#     # 1. Construct the model
#     if MODEL_REGISTRY.get(cfg.MODEL.NAME) is None:
#         model = BaseVideoModel(cfg)
#     else:
#         model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)
#
#     # 2. Check GPU availability
#     if torch.cuda.is_available():
#         assert cfg.NUM_GPUS <= torch.cuda.device_count(), \
#             "Cannot use more GPU devices than available"
#     else:
#         assert cfg.NUM_GPUS == 0, \
#             "Cuda is not available. Please set `NUM_GPUS: 0` for running on CPUs."
#
#     # 3. Move model to primary GPU
#     if cfg.NUM_GPUS:
#         model = model.cuda(gpu_ids[0])  # move to first GPU in list
#
#     # 4. Create EMA model (if enabled)
#     model_ema = None
#     if cfg.MODEL.EMA.ENABLE:
#         model_ema = ModelEmaV2(model, decay=cfg.MODEL.EMA.DECAY)
#
#     # 5. Sync BatchNorm across GPUs (if needed)
#     try:
#         if cfg.BN.SYNC_BN and cfg.NUM_GPUS * cfg.NUM_SHARDS > 1:
#             model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     except Exception as e:
#         logger.warning(f"Failed to apply SyncBatchNorm: {e}")
#
#     # 6. Multi-GPU support
#     if cfg.NUM_GPUS * cfg.NUM_SHARDS > 1:
#         if cfg.PAI:
#             # Distributed training on cluster
#             model = torch.nn.parallel.DistributedDataParallel(model)
#         else:
#             # Use DataParallel for multi-GPU on single machine
#             model = torch.nn.DataParallel(model, device_ids=gpu_ids)
#
#     return model, model_ema

# import sys
# import torch
# import torch.nn as nn
#
# import traceback
#
# import utils.logging as logging
#
# from models.base.models import BaseVideoModel, MODEL_REGISTRY
# from models.utils.model_ema import ModelEmaV2
#
# logger = logging.get_logger(__name__)
#
# def build_model(cfg, gpu_id=None):
#     """
#     Builds the video model.
#     Args:
#         cfg (Config): global config object that provides specifics to construct the model.
#         gpu_id (Optional[int]): specify the gpu index to build model.
#     Returns:
#         model: constructed model
#         model_ema: copied model for ema
#     """
#     # Construct the model
#     gpu_id = cfg.gpu_id
#     if MODEL_REGISTRY.get(cfg.MODEL.NAME) == None:
#         # attempt to find standard models
#         model = BaseVideoModel(cfg)
#     else:
#         # if the model is explicitly defined,
#         # it is directly constructed from the model pool
#         model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)
#
#     if torch.cuda.is_available():
#         assert (
#             cfg.NUM_GPUS <= torch.cuda.device_count()
#         ), "Cannot use more GPU devices than available"
#     else:
#         assert (
#             cfg.NUM_GPUS == 0
#         ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."
#
#     if cfg.NUM_GPUS:
#         if gpu_id is None:
#             # Determine the GPU used by the current process
#             cur_device = torch.cuda.current_device()
#         else:
#             cur_device = gpu_id
#         # model = model.cuda(device=cur_device)
#         model = model.cuda(gpu_id[0])
#     model_ema = None
#     if cfg.MODEL.EMA.ENABLE:
#         model_ema = ModelEmaV2(model, decay=cfg.MODEL.EMA.DECAY)
#
#     try:
#         # convert batchnorm to be synchronized across
#         # different GPUs if needed
#         sync_bn = cfg.BN.SYNC_BN
#         if sync_bn == True and cfg.NUM_GPUS * cfg.NUM_SHARDS > 1:
#             model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     except:
#         sync_bn = None
#
#     # Use multi-process data parallel model in the multi-gpu setting
#     if cfg.NUM_GPUS*cfg.NUM_SHARDS > 1:
#         # Make model replica operate on the current device
#         if cfg.PAI:
#             # Support distributed training on the cluster
#             model = torch.nn.parallel.DistributedDataParallel(
#                 module=model
#             )
#         else:
#             # model = torch.nn.parallel.DistributedDataParallel(
#             #     module=model, device_ids=[cur_device], output_device=cur_device
#             # )
#             model = torch.nn.DataParallel(model, device_ids=cfg.gpu_id)
#
#     return model, model_ema
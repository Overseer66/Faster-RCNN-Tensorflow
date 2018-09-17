# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import pyximport
pyximport.install()

from config import config
if config.USE_GPU_NMS:
    from lib.nms.gpu_nms import gpu_nms
from lib.nms.cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if config.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=config.GPU_ID)
    else:
        return cpu_nms(dets, thresh)

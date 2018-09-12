# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from config import __C
if __C.USE_GPU_NMS:
    from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if __C.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=__C.GPU_ID)
    else:
        return cpu_nms(dets, thresh)

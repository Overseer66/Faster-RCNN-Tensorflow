import numpy as np
from easydict import EasyDict

__C = EasyDict()

config = __C

__C.USE_GPU_NMS = True
__C.GPU_ID = 0

__C.ANCHOR_SCALES = [8, 16, 32]
__C.N_CLASSES = 21

__C.TARGET_SIZE = 1000

__C.TRAIN = EasyDict()

__C.TRAIN.LEARNING_RATE = 1e-3
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.STEPSIZE = 200000

__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

__C.TRAIN.RPN_NMS_THRESHOLD = 0.7
__C.TRAIN.RPN_MIN_SIZE = 16


__C.TRAIN.BATCH_SIZE = 128
__C.TRAIN.FG_FRACTION = 1/4
__C.TRAIN.FG_THRESH = 0.5
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0,1.0,1.0,1.0)



__C.TEST = EasyDict()


__C.TEST.RPN_PRE_NMS_TOP_N = 6000
__C.TEST.RPN_POST_NMS_TOP_N = 300

__C.TEST.RPN_NEGATIVE_OVERLAP = 0.3
__C.TEST.RPN_POSITIVE_OVERLAP = 0.7

__C.TEST.RPN_NMS_THRESHOLD = 0.7



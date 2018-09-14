import numpy as np
import tensorflow as tf
import numpy.random as npr
from config import config as CONFIG
from DeepBuilder.util import safe_append
from generate_anchor import GenerateAnchor
from bbox_transform import BBoxTransform
from bbox_transform import BBoxTransformInverse
from nms_wrapper import nms
from util import ClipBoxes
from util import AnchorOverlaps

def proposal_layer(
        input,
        feature_stride,
        anchor_scales,
        name='ProposalLayer',
        layer_collector=None,
    ):
    with tf.variable_scope(name) as scope:
        l = tf.py_func(
            _proposal_layer,
            [input[0], input[1], input[2], input[3], feature_stride, anchor_scales,],
            tf.float32
        )
        safe_append(layer_collector, l)

        return l


def _proposal_layer(
    rpn_cls_prob,
    rpn_bbox_pred,
    img_info,
    config_key,
    feature_stride=[16,],
    anchor_scales=[8, 16, 32]
    ):
    rpn_cls_prob = np.transpose(rpn_cls_prob, [0, 3, 1, 2])
    rpn_bbox_pred = np.transpose(rpn_bbox_pred, [0, 3, 1, 2])

    img_info = img_info[0]

    config_key = config_key.decode('utf-8')
    config = CONFIG[config_key]

    anchors = GenerateAnchor(scales=np.array(anchor_scales))
    anchors_size = anchors.shape[0]

    scores = rpn_cls_prob[:, anchors_size:, :, :]
    bboxes = rpn_bbox_pred

    height, width = scores.shape[-2:]

    shift_x = np.arange(0, width) * feature_stride
    shift_y = np.arange(0, height) * feature_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    A_size = anchors_size
    K_size = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A_size, 4)) + shifts.reshape((1, K_size, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K_size*A_size, 4))
    
    bboxes = bboxes.transpose((0, 2, 3, 1)).reshape((-1, 4))
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    proposal_bboxes = BBoxTransformInverse(all_anchors, bboxes)
    proposal_bboxes = ClipBoxes(proposal_bboxes, img_info[:2])

    keep_indices = FilterBoxes(proposal_bboxes, config.RPN_MIN_SIZE * config.TARGET_SIZE / np.min(img_info[:2]))
    proposal_bboxes = proposal_bboxes[keep_indices, :]
    scores = scores[keep_indices]

    sorted_indices = scores.ravel().argsort()[::-1]
    if config.RPN_PRE_NMS_TOP_N > 0:
        sorted_indices = sorted_indices[:config.RPN_PRE_NMS_TOP_N]
    proposal_bboxes = proposal_bboxes[sorted_indices, :]
    scores = scores[sorted_indices]

    keep_indices = nms(np.hstack((proposal_bboxes, scores)), config.RPN_NMS_THRESHOLD)
    if config.RPN_POST_NMS_TOP_N > 0:
        keep_indices = keep_indices[:config.RPN_POST_NMS_TOP_N]
    proposal_bboxes = proposal_bboxes[keep_indices, :]
    scores = scores[keep_indices]

    batch_indices = np.zeros((proposal_bboxes.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_indices, proposal_bboxes.astype(np.float32, copy=False)))

    return blob


def FilterBoxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

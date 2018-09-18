import numpy as np
import tensorflow as tf

from config import config as CONFIG
from DeepBuilder.util import safe_append

from lib.FRCNN.bbox_transform import BBoxTransform
from lib.util import AnchorOverlaps


def proposal_target_layer(
        input,
        n_classes,
        name='ProposalTargetLayer',
        layer_collector=None,
    ):
    with tf.variable_scope(name) as scope:
        l = tf.py_func(
            _proposal_target_layer,
            [input[0], input[1], input[2], n_classes,],
            [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        )
        safe_append(layer_collector, l, name)

        return l

def _proposal_target_layer(rpn_rois, gt_boxes, config_key, n_classes):

    config_key = config_key.decode('utf-8')
    config = CONFIG[config_key]

    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    rpn_rois = np.vstack((rpn_rois, np.hstack((zeros, gt_boxes[:, :-1]))))

    n_rois = config.BATCH_SIZE
    n_fg = np.round(config.FG_FRACTION * n_rois)

    rois, labels, bbox_targets, bbox_inside_weights, bbox_target_data = sample_rois(rpn_rois, gt_boxes, n_fg, n_rois, n_classes, config)

    rois = rois.reshape(-1,5)
    labels = labels.reshape(-1,1)
    bbox_targets = bbox_targets.reshape(-1,n_classes*4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1,n_classes*4)
    bbox_outside_weights = np.array(bbox_inside_weights>0).astype(np.float32)

    return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def sample_rois(rpn_rois, gt_boxes, n_fg, n_rois, n_classes, config):

    overlaps = AnchorOverlaps(rpn_rois[:,1:5],gt_boxes[:,:4])
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    fg_inds = np.where(max_overlaps >= config.FG_THRESH)[0]
    n_fg = int(min(n_fg, fg_inds.size))
    if fg_inds.size > 0: fg_inds = np.random.choice(fg_inds, size=n_fg, replace=False)
    bg_inds = np.where((max_overlaps >= config.BG_THRESH_LO) &
                       (max_overlaps < config.BG_THRESH_HI))[0]
    n_bg = n_rois - n_fg
    n_bg = int(min(n_bg, bg_inds.size))
    if bg_inds.size > 0: bg_inds = np.random.choice(bg_inds, size=n_bg, replace=False)

    keep_inds = np.append(fg_inds, bg_inds)
    labels = labels[keep_inds]
    labels[n_fg:] = 0
    rois = rpn_rois[keep_inds]

    bbox_target_data = compute_target(rois[:,1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = get_bbox_regression_labels(bbox_target_data, n_classes, config)



    return rois, labels, bbox_targets, bbox_inside_weights, bbox_target_data


def compute_target(ex, gt, labels):

    targets = BBoxTransform(ex, gt)

    return np.hstack((labels[:,np.newaxis],targets)).astype(np.float32, copy=False)


def get_bbox_regression_labels(bbox_target_data, n_classes, config):

    cls = np.array(bbox_target_data[:,0], dtype=np.uint16, copy=True)
    bbox_targets = np.zeros((cls.size, n_classes*4), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(cls>0)[0]
    for idx in inds:
        start = 4*cls[idx]
        end = start+4
        bbox_targets[idx,start:end] = bbox_target_data[idx,1:]
        bbox_inside_weights[idx,start:end] = config.BBOX_INSIDE_WEIGHTS

    return bbox_targets, bbox_inside_weights
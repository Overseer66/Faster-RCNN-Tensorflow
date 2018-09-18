import numpy as np
import tensorflow as tf

from config import config as CONFIG
from DeepBuilder.util import safe_append

from lib.FRCNN.generate_anchor import GenerateAnchor
from lib.FRCNN.bbox_transform import BBoxTransform
from lib.util import AnchorOverlaps

def split_score_layer(input, shape, name='SplitScore', layer_collector=None, param_collector=None):
    input_shape = tf.shape(input)

    l = tf.transpose(
        tf.reshape(
            tf.transpose(input, [0, 3, 1, 2]),
            [
                input_shape[0],
                int(shape),
                tf.cast(tf.cast(input_shape[1], tf.float32) * (tf.cast(input_shape[3], tf.float32) / tf.cast(shape, tf.float32)), tf.int32),
                input_shape[2]
            ]
        ),
        [0, 2, 3, 1],
        name=name
    )

    safe_append(layer_collector, l, name)

    return l


def combine_score_layer(input, shape, name='CombineScore', layer_collector=None, param_collector=None):
    input_shape = tf.shape(input)

    l = tf.transpose(
        tf.reshape(
            tf.transpose(input,[0, 3, 1, 2]),
            [
                input_shape[0],
                int(shape),
                tf.cast(tf.cast(input_shape[1], tf.float32) / tf.cast(shape, tf.float32) * tf.cast(input_shape[3], tf.float32), tf.int32),
                input_shape[2]
            ]
        ),
        [0, 2, 3, 1],
        name=name
    )

    safe_append(layer_collector, l, name)

    return l


def anchor_target_layer(
        input,
        feature_stride,
        anchor_scales,
        name='AnchorLayer',
        layer_collector=None,
    ):
    with tf.variable_scope(name) as scope:
        layers = tf.py_func(
            _anchor_target_layer,
            [input[0], input[1], input[2], input[3], feature_stride, anchor_scales,],
            [tf.float32, tf.float32, tf.float32, tf.float32]
        )
        for layer in layers:
            safe_append(layer_collector, layer)

        return layers


def _anchor_target_layer(
    rpn_cls_score_layer,
    img_info,
    gt_boxes,
    config_key,
    feature_stride=[16,],
    anchor_scales=[8, 16, 32]
    ):
    img_info = img_info[0]

    config_key = config_key.decode('utf-8')
    config = CONFIG[config_key]

    anchors = GenerateAnchor(scales=np.array(anchor_scales))

    height, width = rpn_cls_score_layer.shape[1:3]

    shift_x = np.arange(0, width) * feature_stride
    shift_y = np.arange(0, height) * feature_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    A_size = len(anchors)
    K_size = len(shifts)
    all_anchors = (anchors.reshape((1, A_size, 4)) + shifts.reshape((1, K_size, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K_size*A_size, 4))
    all_anchors_size = len(all_anchors)

    allowed_border = 0

    inside_indices = np.where(
        (all_anchors[:, 0] >= -allowed_border) &
        (all_anchors[:, 1] >= -allowed_border) &
        (all_anchors[:, 2] < img_info[1] + allowed_border) & # Width
        (all_anchors[:, 3] < img_info[0] + allowed_border)   # Height
    )[0]

    all_anchors = all_anchors[inside_indices]

    labels = np.empty((len(inside_indices),), dtype=np.float32)
    labels.fill(-1)

    overlaps = AnchorOverlaps(all_anchors, gt_boxes)

    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inside_indices)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps==gt_max_overlaps)[0]

    labels[max_overlaps < config.RPN_NEGATIVE_OVERLAP] = 0
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= config.RPN_POSITIVE_OVERLAP] = 1

    bbox_targets = ComputTargets(all_anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inside_indices), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0, 1.0, 1.0))

    bbox_outside_weights = np.zeros((len(inside_indices), 4), dtype=np.float32)
    if -1.0 < 0:
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        positive_weights = -1.0 / np.sum(labels == 1)
        negative_weights = (1.0 - -1.0) / np.sum(labels == 0)
    bbox_outside_weights[labels==1, :] = positive_weights
    bbox_outside_weights[labels==0, :] = negative_weights


    labels = Unmap(labels, all_anchors_size, inside_indices, fill=-1)
    bbox_targets = Unmap(bbox_targets, all_anchors_size, inside_indices, fill=0)
    bbox_inside_weights = Unmap(bbox_inside_weights, all_anchors_size, inside_indices, fill=0)
    bbox_outside_weights = Unmap(bbox_outside_weights, all_anchors_size, inside_indices, fill=0)

    labels = labels.reshape((1, height, width, A_size)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A_size*height, width))
    rpn_labels = labels

    bbox_targets = bbox_targets.reshape((1, height, width, A_size*4)).transpose(0, 3, 1, 2)
    rpn_bbox_targets = bbox_targets

    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A_size*4)).transpose(0, 3, 1, 2)
    rpn_bbox_inside_weights = bbox_inside_weights

    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A_size*4)).transpose(0, 3, 1, 2)
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def ComputTargets(ex_rois, gt_rois):
    print(ex_rois.shape, gt_rois.shape)
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return BBoxTransform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def Unmap(data, count, indices, fill=0):
    if len(data.shape) == 1:
        retval = np.empty((count, ), dtype=np.float32)
        retval.fill(fill)
        retval[indices] = data
    else:
        retval = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        retval.fill(fill)
        retval[indices, :] = data

    return retval
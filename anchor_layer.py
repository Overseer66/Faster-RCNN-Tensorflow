import numpy as np
import tensorflow as tf
from DeepBuilder.util import *


def anchor_layer(
        input,
        feature_stride,
        anchor_scales,
        name='_AnchorLayer',
        layer_collector=None,
        param_collector=None,
):
    layers = tf.py_func(
        _anchor_layer,
        [input[0], input[1], input[2], feature_stride, anchor_scales,],
        [tf.float32, tf.float32, tf.float32, tf.float32]
    )
    for layer in layers:
        safe_append(layer_collector, layer)

    return layers


def _anchor_layer(rpn_cls_score_layer, img_info, gt_boxes, feature_stride=[16,], anchor_scales=[8, 16, 32]):
    img_info = img_info[0]

    anchors = generate_anchor(scales=np.array(anchor_scales))

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

    org_anchors = all_anchors
    all_anchors = all_anchors[inside_indices]

    labels = np.empty((len(inside_indices),), dtype=np.float32)
    labels.fill(-1)

    overlaps = AnchorOverlaps(all_anchors, gt_boxes)

    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inside_indices)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps==gt_max_overlaps)[0]

    labels[max_overlaps < 0.3] = 0
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= 0.7] = 1

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



def generate_anchor(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    base_anchor = np.array([1, 1, base_size, base_size]) - 1

    w, h, x_ctr, y_ctr = Width_Height_Xctr_Yctr(base_anchor)
    wh_size = w * h
    size_ratios = wh_size / ratios

    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = MakeAnchors(ws, hs, x_ctr, y_ctr)

    anchors = np.vstack([AnchorScales(anchor, scales) for anchor in anchors])

    return anchors

def Width_Height_Xctr_Yctr(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)

    return w, h, x_ctr, y_ctr

def MakeAnchors(Widths, Heights, Xctr, Yctr):
    Widths = Widths[:, np.newaxis]
    Heights = Heights[:, np.newaxis]
    anchors = np.hstack((
        Xctr - 0.5 * (Widths - 1),
        Yctr - 0.5 * (Heights - 1),
        Xctr + 0.5 * (Widths - 1),
        Yctr + 0.5 * (Heights - 1),
    ))

    return anchors

def AnchorScales(anchor, scales):
    w, h, x_ctr, y_ctr = Width_Height_Xctr_Yctr(anchor)
    ws = w * scales
    hs = h * scales
    anchors = MakeAnchors(ws, hs, x_ctr, y_ctr)

    return anchors


def AnchorOverlaps(anchors, gt_boxes):
    A_size = len(anchors)
    K_size = len(gt_boxes)

    overlaps = np.zeros((A_size, K_size), dtype=np.float32)
    for k in range(K_size):
        box_area = (
            (gt_boxes[k, 2] - gt_boxes[k, 0] + 1) * # Width Size
            (gt_boxes[k, 3] - gt_boxes[k, 1] + 1)   # Height Size
        )
        for a in range(A_size):
            iw = min(anchors[a, 2], gt_boxes[k, 2]) - max(anchors[a, 0], gt_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(anchors[a, 3], gt_boxes[k, 3]) - max(anchors[a, 1], gt_boxes[k, 1]) + 1
                if ih > 0:
                    ua = (anchors[a, 2] - anchors[a, 0] + 1) * (anchors[a, 3] - anchors[a, 1] + 1) + box_area - (iw * ih)

                    overlaps[a, k] = iw * ih / ua
    return overlaps


def ComputTargets(ex_rois, gt_rois):
    print(ex_rois.shape, gt_rois.shape)
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return BBoxTransform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def BBoxTransform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


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
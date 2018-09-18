import tensorflow as tf
import numpy as np
import os

def find_path(dirs, filename):
    for dir in dirs:
        filepath = os.path.join(dir, filename)
        if os.path.exists(filepath):
            return filepath


def ClipBoxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


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


def ModifiedSmoothL1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):

    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

    return outside_mul
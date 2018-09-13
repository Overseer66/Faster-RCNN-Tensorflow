import numpy as np

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

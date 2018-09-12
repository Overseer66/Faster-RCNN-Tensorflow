import numpy as np
import tensorflow as tf

from DeepBuilder import layer, activation, build
from anchor_layer import anchor_layer
from rpn_layers import split_score, combine_score


vgg16 = (
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 64],}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 64],}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 64],}},
    {'method': layer.max_pool, 'args': (), 'kwargs': {'padding': 'VALID',}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 128],}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 128],}},
    {'method': layer.max_pool, 'args': (), 'kwargs': {'padding': 'VALID',}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 256],}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 256],}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 256],}},
    {'method': layer.max_pool, 'args': (), 'kwargs': {'padding': 'VALID',}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.max_pool, 'args': (), 'kwargs': {'padding': 'VALID',}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
)

anchor_scales = [8, 16, 32]

rpn_conv = (
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
)
rpn_bbox = (
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*4], 'padding': 'VALID', 'activation': None}},
)
rpn_score = (
    {'method': layer.conv_2d, 'args': (), 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*2], 'padding': 'VALID', 'activation': None}},
)
rpn_data = (
    {'method': anchor_layer, 'args': (), 'kwargs': {'feature_stride': [16,], 'anchor_scales': [8, 16, 32]}},
)

rpn_cls_prob = (
    {'method': split_score, 'args': (), 'kwargs': {'shape': [9, 2]}},
    {'method': activation.Softmax, 'args': (), 'kwargs': {}},
    {'method': combine_score, 'args': (), 'kwargs': {'shape': 9*2}},
)

if __name__ == '__main__':
    Image = tf.placeholder(tf.float32, [None, None, None, 3])

    VGG16_Builder = build.Builder(vgg16)
    VGG16_LastLayer, VGG16_Layers, VGG16_Params = VGG16_Builder(Image, 'VGG16', reuse=False)

    RPN_Builder = build.Builder(rpn_conv)
    RPN, RPN_Layers, RPN_Params = RPN_Builder(VGG16_LastLayer, 'RPN', reuse=False)

    RPN_BBox_Builder = build.Builder(rpn_bbox)
    RPN_BBox, RPN_BBox_Layers, RPN_BBox_Params = RPN_BBox_Builder(RPN, 'RPN_BBOX', reuse=False)

    RPN_BBox_Score_Builder = build.Builder(rpn_score)
    RPN_BBox_Score, RPN_BBox_Score_Layers, RPN_BBox_Score_Params = RPN_BBox_Score_Builder(RPN, 'RPN_BBOX_SCORE', reuse=False)

    RPN_CLS_Prob_Builder = build.Builder(rpn_cls_prob)
    RPN_CLS_Prob, RPN_CLS_Prob_Layers, RPN_CLS_Prob_Params = RPN_CLS_Prob_Builder(RPN_BBox_Score, 'RPN_CLS_PROB', reuse=False)

    ImageInfo = tf.placeholder(tf.float32, [None, 3])
    GroundTruth = tf.placeholder(tf.float32, [None, 5])

    RPN_Data_Builder = build.Builder(rpn_data)
    _tensors, RPN_Data_Layers, RPN_Data_Params = RPN_Data_Builder([RPN_BBox_Score, ImageInfo, GroundTruth], 'RPN_DATA', reuse=False)
    RPN_Labels, RPN_BBox_Targets, RPN_BBox_Inside_Weights, RPN_BBox_Outside_Weights = _tensors


    img_wsize = 256
    img_hsize = 256
    img = np.random.rand(1, img_hsize, img_wsize, 3)
    img_info = np.random.rand(1, 3)
    img_info[:, 0] = img_hsize
    img_info[:, 1] = img_wsize
    gt_boxes = np.random.rand(2, 5)
    gt_boxes[0, [0, 2]] = np.array([0.2, 0.6]) * img_wsize # L R
    gt_boxes[0, [1, 3]] = np.array([0.3, 0.8]) * img_hsize # T B
    gt_boxes[1, [0, 2]] = np.array([0.4, 0.6]) * img_wsize
    gt_boxes[1, [1, 3]] = np.array([0.7, 0.8]) * img_hsize

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run(session=sess)
    result = sess.run(
        [RPN_CLS_Prob],
        {
            Image: img,
            ImageInfo: img_info,
            GroundTruth: gt_boxes
        }
    )

    pass





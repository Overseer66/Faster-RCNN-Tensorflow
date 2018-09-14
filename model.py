import numpy as np
import tensorflow as tf

from config import config as CONFIG
from DeepBuilder import layer, activation, build
from anchor_layer import anchor_target_layer, split_score_layer, combine_score_layer
from proposal_layer import proposal_layer, proposal_target_layer
from roi_layer import roi_pooling
from data_importer import import_image_and_xml
from data_importer import get_class_idx


vgg16 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64],}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64],}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64],}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID',}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128],}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128],}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID',}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256],}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256],}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256],}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID',}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID',}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
)

anchor_scales = [8, 16, 32]

rpn_conv = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512],}},
)
rpn_bbox = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*4], 'padding': 'VALID', 'activation': None}},
)
rpn_score = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*2], 'padding': 'VALID', 'activation': None}},
)
rpn_data = (
    {'method': anchor_target_layer, 'kwargs': {'feature_stride': [16,], 'anchor_scales': [8, 16, 32]}},
)

rpn_cls_prob = (
    {'method': split_score_layer, 'kwargs': {'shape': 2}},
    {'method': activation.Softmax},
    {'method': combine_score_layer, 'kwargs': {'shape': len(anchor_scales)*3*2}},
)

rpn_proposals = (
    {'method': proposal_layer, 'kwargs': {'feature_stride': [16,], 'anchor_scales': [8, 16, 32]}},
)

roi_data = (
    {'method': proposal_target_layer, 'kwargs': {'n_classes': 21}},
)

roi_pool = (
    {'method': roi_pooling, 'kwargs': {'pooled_width': 7, 'pooled_height': 7, 'spatial_scale': 1.0/16}},
    {'method': activation.Transpose, 'kwargs': {'permutation': [0, 3, 1, 2]}},
    {'method': layer.flatten},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 4096}},
    {'method': activation.Dropout, 'kwargs': {'keep_prob': 0.5}},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 4096}},
    {'method': activation.Dropout, 'kwargs': {'keep_prob': 0.5}},
)

pred_score = (
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 21, 'activation': None}},
    {'method': activation.Softmax}
)

pred_bbox = (
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 21*4, 'activation': None}},
)


if __name__ == '__main__':
    Image = tf.placeholder(tf.float32, [None, None, None, 3])

    VGG16_Builder = build.Builder(vgg16)
    VGG16_LastLayer, VGG16_Layers, VGG16_Params = VGG16_Builder(Image, 'VGG16')

    RPN_Builder = build.Builder(rpn_conv)
    RPN, RPN_Layers, RPN_Params = RPN_Builder(VGG16_LastLayer, 'RPN')

    RPN_BBox_Builder = build.Builder(rpn_bbox)
    RPN_BBox, RPN_BBox_Layers, RPN_BBox_Params = RPN_BBox_Builder(RPN, 'RPN_BBOX')

    RPN_BBox_Score_Builder = build.Builder(rpn_score)
    RPN_BBox_Score, RPN_BBox_Score_Layers, RPN_BBox_Score_Params = RPN_BBox_Score_Builder(RPN, 'RPN_BBOX_SCORE')

    RPN_CLS_Prob_Builder = build.Builder(rpn_cls_prob)
    RPN_CLS_Prob, RPN_CLS_Prob_Layers, RPN_CLS_Prob_Params = RPN_CLS_Prob_Builder(RPN_BBox_Score, 'RPN_CLS_PROB')

    ImageInfo = tf.placeholder(tf.float32, [None, 3])
    GroundTruth = tf.placeholder(tf.float32, [None, 5])

    RPN_Data_Builder = build.Builder(rpn_data)
    _tensors, RPN_Data_Layers, RPN_Data_Params = RPN_Data_Builder([RPN_BBox_Score, ImageInfo, GroundTruth, 'TRAIN'], 'RPN_DATA')
    RPN_Labels, RPN_BBox_Targets, RPN_BBox_Inside_Weights, RPN_BBox_Outside_Weights = _tensors

    RPN_Proposals_Builder = build.Builder(rpn_proposals)
    RPN_Proposals, RPN_Proposals_Layer, RPN_Proposals_Params = RPN_Proposals_Builder([RPN_CLS_Prob, RPN_BBox, ImageInfo, 'TRAIN'], 'RPN_PROPOSALS')

    ROI_Data_Builder = build.Builder(roi_data)
    ROI_Data, ROI_Data_Layer, RPN_ROI_Data_Params = ROI_Data_Builder([RPN_Proposals, GroundTruth, 'TRAIN'], 'ROI_DATA')
    # ROI Target Layer Build

    ROI_Pool_Builder = build.Builder(roi_pool)
    ROI_Pool, ROI_Pool_Layer, ROI_Pool_Params = ROI_Pool_Builder([VGG16_LastLayer, RPN_Proposals], 'ROI_POOLING')

    Pred_Score_Builder = build.Builder(pred_score)
    Pred_Score, Pred_Score_Layer, Pred_Score_Params = Pred_Score_Builder(ROI_Pool, 'PRED_SCORE')

    Pred_BBox_Builder = build.Builder(pred_bbox)
    Pred_BBox, Pred_BBox_Layer, Pred_BBox_Params = Pred_BBox_Builder(ROI_Pool, 'PRED_BBOX')

    images, xmls = import_image_and_xml('./data/sample_jpg/', './data/sample_xml/')

    img_idx = 3
    img = images[img_idx]
    img_wsize = img.shape[0]
    img_hsize = img.shape[1]
    img = [img]
    img_info = np.array([[img_hsize, img_wsize, CONFIG.TRAIN.TARGET_SIZE/min(img_wsize, img_hsize)]])
    gt_boxes = [xmls['boxes'][img_idx][0] + [get_class_idx(xmls['classes'][img_idx][0])]]

    # img_wsize = 256
    # img_hsize = 256
    # img = np.random.rand(1, img_hsize, img_wsize, 3)
    # img_info = np.random.rand(1, 3)
    # img_info[:, 0] = img_hsize
    # img_info[:, 1] = img_wsize
    # img_info[:, 2] = CONFIG.TRAIN.TARGET_SIZE / min(img_wsize, img_hsize)
    # gt_boxes = np.random.rand(2, 5)
    # gt_boxes[0, [0, 2]] = np.array([0.2, 0.6]) * img_wsize # L R
    # gt_boxes[0, [1, 3]] = np.array([0.3, 0.8]) * img_hsize # T B
    # gt_boxes[1, [0, 2]] = np.array([0.4, 0.6]) * img_wsize
    # gt_boxes[1, [1, 3]] = np.array([0.7, 0.8]) * img_hsize

    ConfigProto = tf.ConfigProto(allow_soft_placement=True)
    ConfigProto.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=ConfigProto)
    tf.global_variables_initializer().run(session=sess)
    result = sess.run(
        [ROI_Data],
        {
            Image: img,
            ImageInfo: img_info,
            GroundTruth: gt_boxes
        }
    )

    # print([result[0][i].shape for i in range(len(result[0]))])

    pass






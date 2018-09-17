import numpy as np
import tensorflow as tf
import cv2

from config import config as CONFIG
from DeepBuilder import layer, activation, build
from lib.FRCNN.anchor_layer import anchor_target_layer, split_score_layer, combine_score_layer
from lib.FRCNN.proposal_layer import proposal_layer
from lib.FRCNN.proposal_target_layer import proposal_target_layer
from lib.FRCNN.bbox_transform import BBoxTransformInverse
from lib.roi_layer import roi_pooling
from lib.nms_wrapper import nms
from lib.util import ClipBoxes

from data_importer import import_image_and_xml, get_class_idx, get_class_name

vgg16 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'conv1_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'conv1_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': 'conv2_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': 'conv2_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': 'conv3_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': 'conv3_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': 'conv3_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv4_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv4_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv4_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool4'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv5_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv5_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv5_3'}},
)

anchor_scales = [8, 16, 32]
n_classes = 21

rpn_conv = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'rpn_conv/3x3'}},
)
rpn_bbox = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*4], 'padding': 'VALID', 'activation': None, 'name': 'rpn_bbox_pred'}},
)
rpn_score = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*2], 'padding': 'VALID', 'activation': None, 'name': 'rpn_cls_score'}},
)
rpn_data = (
    {'method': anchor_target_layer, 'kwargs': {'feature_stride': [16,], 'anchor_scales': [8, 16, 32]}},
)
rpn_cls_prob = (
    {'method': split_score_layer, 'kwargs': {'shape': 2}},
    {'method': activation.Softmax, 'kwargs': {'name': 'rpn_cls_prob'}},
    {'method': combine_score_layer, 'kwargs': {'shape': len(anchor_scales)*3*2}},
)
rpn_proposals = (
    {'method': proposal_layer, 'kwargs': {'feature_stride': [16,], 'anchor_scales': [8, 16, 32], 'name': 'rois'}},
    {'method': layer.reshape, 'kwargs': {'shape': [-1, 5]}},
)

roi_data = (
    {'method': proposal_target_layer, 'kwargs': {'n_classes': 21}},
)
roi_bbox = (
    {'method': layer.reshape, 'kwargs': {'shape': [-1, 5]}},
)
roi_pool = (
    {'method': roi_pooling, 'kwargs': {'pooled_width': 7, 'pooled_height': 7, 'spatial_scale': 1.0/16}, 'name': 'pool_5'},
    {'method': activation.Transpose, 'kwargs': {'permutation': [0, 3, 1, 2]}},
    {'method': layer.flatten},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 4096, 'name': 'fc6'}},
    #{'method': activation.Dropout, 'kwargs': {'keep_prob': 0.5}},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 4096, 'name': 'fc7'}},
    #{'method': activation.Dropout, 'kwargs': {'keep_prob': 0.5}},
)

pred_score = (
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': n_classes, 'activation': None, 'name': 'cls_score'}},
    {'method': activation.Softmax, 'name': 'cls_prob'}
)
pred_bbox = (
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': n_classes*4, 'activation': None, 'name': 'bbox_pred'}},
)



Image = tf.placeholder(tf.float32, [None, None, None, 3])

VGG16_Builder = build.Builder(vgg16)
VGG16_LastLayer, VGG16_Layers, VGG16_Params = VGG16_Builder(Image)

RPN_Builder = build.Builder(rpn_conv)
RPN, RPN_Layers, RPN_Params = RPN_Builder(VGG16_LastLayer)

RPN_BBox_Builder = build.Builder(rpn_bbox)
RPN_BBox, RPN_BBox_Layers, RPN_BBox_Params = RPN_BBox_Builder(RPN)

RPN_BBox_Score_Builder = build.Builder(rpn_score)
RPN_BBox_Score, RPN_BBox_Score_Layers, RPN_BBox_Score_Params = RPN_BBox_Score_Builder(RPN)

ImageInfo = tf.placeholder(tf.float32, [None, 3])
GroundTruth = tf.placeholder(tf.float32, [None, 5])

RPN_Data_Builder = build.Builder(rpn_data)
_tensors, RPN_Data_Layers, RPN_Data_Params = RPN_Data_Builder([RPN_BBox_Score, ImageInfo, GroundTruth, 'TRAIN'])
RPN_Labels, RPN_BBox_Targets, RPN_BBox_Inside_Weights, RPN_BBox_Outside_Weights = _tensors

RPN_CLS_Prob_Builder = build.Builder(rpn_cls_prob)
RPN_CLS_Prob, RPN_CLS_Prob_Layers, RPN_CLS_Prob_Params = RPN_CLS_Prob_Builder(RPN_BBox_Score)

RPN_Proposals_Builder = build.Builder(rpn_proposals)
RPN_Proposals, RPN_Proposals_Layer, RPN_Proposals_Params = RPN_Proposals_Builder([RPN_CLS_Prob, RPN_BBox, ImageInfo, 'TRAIN'])

ROI_Data_Builder = build.Builder(roi_data)
_tensors, ROI_Data_Layer, RPN_ROI_Data_Params = ROI_Data_Builder([RPN_Proposals, GroundTruth, 'TRAIN'])
ROI_BBox, ROI_Labels, ROI_BBox_Targets, ROI_BBox_Inside_Weights, ROI_BBox_Outside_Weights = _tensors

ROI_BBox_Builder = build.Builder(roi_bbox)
ROI_BBox, ROI_BBox_Layer, ROI_BBox_Params = ROI_BBox_Builder(ROI_BBox)

ROI_Pool_Builder = build.Builder(roi_pool)
#ROI_Pool, ROI_Pool_Layer, ROI_Pool_Params = ROI_Pool_Builder([VGG16_LastLayer, ROI_BBox], 'ROI_POOLING')
ROI_Pool, ROI_Pool_Layer, ROI_Pool_Params = ROI_Pool_Builder([VGG16_LastLayer, RPN_Proposals])

Pred_Score_Builder = build.Builder(pred_score)
Pred_Prob, Pred_Score_Layer, Pred_Score_Params = Pred_Score_Builder(ROI_Pool)

Pred_BBox_Builder = build.Builder(pred_bbox)
Pred_BBox, Pred_BBox_Layer, Pred_BBox_Params = Pred_BBox_Builder(ROI_Pool)



if __name__ == '__main__':
    # images, xmls = import_image_and_xml('./data/sample_jpg/', './data/sample_xml/')
    # img_idx = 1
    # img = images[img_idx]
    img_org = cv2.imread('data/sample_jpg/2007_000027.jpg')
    img = img_org

    img_wsize = img.shape[1]
    img_hsize = img.shape[0]
    
    img_min_size = min(img_wsize, img_hsize)
    img_scale = CONFIG.TRAIN.TARGET_SIZE / img_min_size

    img = img.astype(np.float32)
    img -= CONFIG.PIXEL_MEANS

    img = cv2.resize(img, None, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)


    img_wsize = img.shape[1]
    img_hsize = img.shape[0]
    img = [img]
    img_info = np.array([[img_hsize, img_wsize, img_scale]])
    # gt_boxes = [xmls['boxes'][img_idx][0] + [get_class_idx(xmls['classes'][img_idx][0])]]

    ConfigProto = tf.ConfigProto(allow_soft_placement=True)
    ConfigProto.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=ConfigProto)
    
    #tf.global_variables_initializer().run(session=sess)
    saver = tf.train.Saver()
    saver.restore(sess, 'data/pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt')

    pred_bbox, pred_prob, rois, test = sess.run(
        [Pred_BBox, Pred_Prob, RPN_Proposals, VGG16_LastLayer],
        {
            Image: img,
            ImageInfo: img_info,
            #GroundTruth: gt_boxes
        }
    )

    boxes = rois[:, 1:5] / img_scale
    pred_boxes = pred_bbox
    pred_boxes = BBoxTransformInverse(boxes, pred_boxes)
    pred_boxes = ClipBoxes(pred_boxes, (img_hsize, img_wsize))

    import matplotlib.pyplot as plt

    def vis_detections(im, class_name, dets,ax, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='red', linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title(('{} detections with '
                    'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                    thresh),
                    fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()

    img = img_org
    img = img[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    for idx in range(n_classes-1):
        idx += 1
        cls_boxes = pred_boxes[:, 4*idx:4*(idx+1)]
        cls_scores = pred_prob[:, idx]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, 0.3)
        dets = dets[keep, :]
        vis_detections(img, get_class_name(idx-1), dets, ax)

    plt.show()

    pass






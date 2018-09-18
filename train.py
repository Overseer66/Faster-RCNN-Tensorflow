import cv2
import numpy as np

from config import config as CONFIG
from lib.data.voc_importer import *

from DeepBuilder.util import SearchLayer
# from DeepBuilder.util import ModifiedSmoothL1

from architecture.vgg import *
from architecture.rpn import *
from architecture.roi import *

from lib.FRCNN.bbox_transform import BBoxTransformInverse

Image = tf.placeholder(tf.float32, [None, None, None, 3], name='image')
ImageInfo = tf.placeholder(tf.float32, [None, 3], name='image_info')
GroundTruth = tf.placeholder(tf.float32, [None, 5], name='ground_truth')
ConfigKey = tf.placeholder(tf.string, name='config_key')

VGG16_Builder = build.Builder(vgg16)
VGG16_LastLayer, VGG16_Layers, VGG16_Params = VGG16_Builder(Image)

# Train Model
RPN_Builder = build.Builder(rpn_train)
RPN_Proposal_BBoxes, RPN_Layers, RPN_Params = RPN_Builder([ImageInfo, GroundTruth, ConfigKey, VGG16_LastLayer])

ROI_Builder = build.Builder(roi_train)
Pred_BBoxes, ROI_Layers, ROI_Params = ROI_Builder([VGG16_LastLayer, RPN_Proposal_BBoxes, GroundTruth, ConfigKey])
Pred_CLS_Prob = SearchLayer(ROI_Layers, 'cls_prob')

# LOSS
# RPN_class
rpn_anchor_target = SearchLayer(RPN_Layers, 'anchor_target_layer')
rpn_cls_score = SearchLayer(RPN_Layers, 'rpn_cls_score_reshape')
rpn_cls_score = tf.reshape(rpn_cls_score, [-1,2])
rpn_cls_label = tf.reshape(rpn_anchor_target[0],[-1])
rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_cls_label,-1))),[-1,2])
rpn_cls_label = tf.reshape(tf.gather(rpn_cls_label,tf.where(tf.not_equal(rpn_cls_label,-1))),[-1])
rpn_cls_label = tf.cast(rpn_cls_label, dtype=tf.int32)
rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_cls_label))

# RPN_bbox
rpn_bbox_pred = SearchLayer(RPN_Layers, 'rpn_bbox_pred')
rpn_bbox_target = tf.transpose(rpn_anchor_target[1],[0,2,3,1])
rpn_bbox_inside = tf.transpose(rpn_anchor_target[2],[0,2,3,1])
rpn_bbox_outside = tf.transpose(rpn_anchor_target[3],[0,2,3,1])
rpn_bbox_l1 = ModifiedSmoothL1(3.0, rpn_bbox_pred, rpn_bbox_target, rpn_bbox_inside, rpn_bbox_outside)
rpn_bbox_loss = tf.reduce_mean(tf.reduce_sum(rpn_bbox_l1, reduction_indices=[1, 2, 3]))

# RCNN_class
rcnn_proposal_target = SearchLayer(ROI_Layers, 'proposal_target_layer')
rcnn_cls_score = SearchLayer(ROI_Layers, 'cls_score')
rcnn_cls_label = rcnn_proposal_target[1]
rcnn_cls_label = tf.cast(tf.reshape(rcnn_cls_label, [-1]), dtype=tf.int32)
rcnn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rcnn_cls_score, labels=rcnn_cls_label))

# RCNN_bbox
rcnn_bbox_pred = SearchLayer(ROI_Layers, 'bbox_pred')
rcnn_bbox_target = rcnn_proposal_target[2]
rcnn_bbox_inside = rcnn_proposal_target[3]
rcnn_bbox_outside = rcnn_proposal_target[4]
rcnn_bbox_l1 = ModifiedSmoothL1(1.0, rcnn_bbox_pred, rcnn_bbox_target, rcnn_bbox_inside, rcnn_bbox_outside)
rcnn_bbox_loss = tf.reduce_mean(tf.reduce_sum(rcnn_bbox_l1, reduction_indices=[1]))

final_loss = rpn_cls_loss + rpn_bbox_loss + rcnn_cls_loss + rcnn_bbox_loss

# global_step = tf.Variable(0, trainable=False)
# lr = tf.train.exponential_decay(CONFIG.TRAIN.LEARNING_RATE, global_step, CONFIG.TRAIN.STEPSIZE, 0.1, staircase=True)
# momentum = CONFIG.TRAIN.MOMENTUM
# train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(final_loss, global_step=global_step)



if __name__ == '__main__':
    images, xmls = import_image_and_xml('./data/sample_jpg/', './data/sample_xml/')
    idx = 3
    img_org = images[idx]
    img = img_org
    gt_boxes = [xmls['boxes'][idx][0] + [get_class_idx(xmls['classes'][idx][0])]]
    # img_org = cv2.imread('data/sample_jpg/2007_000027.jpg')
    # img = img_org

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

    # tf.global_variables_initializer().run(session=sess)
    saver = tf.train.Saver()
    saver.restore(sess, 'data/pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt')

    rpn_cls_loss_v, rpn_bbox_loss_v, rcnn_cls_loss_v, rcnn_bbox_loss_v = sess.run(
        [rpn_cls_loss, rpn_bbox_loss, rcnn_cls_loss, rcnn_bbox_loss],
        {
            Image: img,
            ImageInfo: img_info,
            GroundTruth: gt_boxes,
            ConfigKey: 'TRAIN',
        }
    )

    print(rpn_cls_loss_v, rpn_bbox_loss_v, rcnn_cls_loss_v, rcnn_bbox_loss_v)

    pass

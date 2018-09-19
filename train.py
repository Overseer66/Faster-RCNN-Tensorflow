import cv2
import numpy as np
import time

from config import config as CONFIG
from lib.data.voc_importer import *

from DeepBuilder.util import SearchLayer
from lib.util import ModifiedSmoothL1

from architecture.vgg import *
from architecture.rpn import *
from architecture.roi import *

from lib.data.voc_importer import *
from lib.data.util import ImageSetExpand

Image = tf.placeholder(tf.float32, [None, None, None, 3], name='image')
ImageInfo = tf.placeholder(tf.float32, [None, 3], name='image_info')
GroundTruth = tf.placeholder(tf.float32, [None, 5], name='ground_truth')
ConfigKey = tf.placeholder(tf.string, name='config_key')

VGG16_Builder = build.Builder(vgg16)
VGG16_LastLayer, VGG16_Layers, VGG16_Params = VGG16_Builder(Image)

# Train Model
RPN_Builder = build.Builder(rpn_train)
RPN_Proposal_BBoxes, RPN_Layers, RPN_Params = RPN_Builder([[ImageInfo, GroundTruth, ConfigKey, VGG16_LastLayer], ['image_info', 'ground_truth', 'config_key', 'conv5_3']])

ROI_Builder = build.Builder(roi_train)
Pred_BBoxes, ROI_Layers, ROI_Params = ROI_Builder([[VGG16_LastLayer, RPN_Proposal_BBoxes, GroundTruth, ConfigKey], ['conv5_3', 'rpn_proposal_bboxes', 'ground_truth', 'config_key']])
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

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(CONFIG.TRAIN.LEARNING_RATE, global_step, CONFIG.TRAIN.STEPSIZE, 0.1, staircase=True)
momentum = CONFIG.TRAIN.MOMENTUM
train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(final_loss, global_step=global_step)

def get_class_idx(name):
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    return class_names.index(name)+1

if __name__ == '__main__':
    org_image_set = voc_xml_parser('./data/sample_jpg/', './data/sample_xml/')
    image_set = ImageSetExpand(org_image_set)

    ConfigProto = tf.ConfigProto(allow_soft_placement=True)
    ConfigProto.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=ConfigProto)

    tf.global_variables_initializer().run(session=sess)
    saver = tf.train.Saver()
    # saver.restore(sess, 'data/pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt')

    for rpt in range(5000):
        for idx, (img, img_info, gt_boxes, gt_classes) in enumerate(zip(image_set['images'], image_set['image_shape'], image_set['boxes'], image_set['classes'])):
            gts = [np.concatenate([gt_boxes[i], [get_class_idx(gt_classes[i])]]) for i in range(len(gt_boxes))]

            start_time = time.time()
            rpn_cls_loss_v, rpn_bbox_loss_v, rcnn_cls_loss_v, rcnn_bbox_loss_v, _, rcnn_score, rcnn_label, rpn_score, rpn_label  = sess.run(
                [rpn_cls_loss, rpn_bbox_loss, rcnn_cls_loss, rcnn_bbox_loss, train_op, rcnn_cls_score, rcnn_cls_label, rpn_cls_score, rpn_cls_label],
                {
                    Image: [img],
                    ImageInfo: [img_info],
                    GroundTruth: gts,
                    ConfigKey: 'TRAIN',
                }
            )
            end_time = time.time()

            # print(rpn_cls_loss_v, rpn_bbox_loss_v, rcnn_cls_loss_v, rcnn_bbox_loss_v)
            print("Step", rpt*len(image_set['classes'])+idx)
            print("Loss :", rpn_cls_loss_v,'\t',rpn_bbox_loss_v,'\t',rcnn_cls_loss_v,'\t',rcnn_bbox_loss_v)
            print("Total Loss :", rpn_cls_loss_v+rpn_bbox_loss_v+rcnn_cls_loss_v+rcnn_bbox_loss_v)
            print('Figure %2d Recognition done. - %5.2f (s)' % (idx+1, end_time-start_time))

        if rpt%1000 == 0:
            saver.save(sess, './data/pretrain_model/sample.ckpt', global_step=global_step)


    pass

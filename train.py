import cv2
import numpy as np
import time
import sys
import os

from config import config as CONFIG

from DeepBuilder.util import SearchLayer
from lib.util import ModifiedSmoothL1

from architecture.vgg import *
from architecture.inception_v2 import *
from architecture.inception_v4 import *
from architecture.mobilenet import *
from architecture.resnet import *
from architecture.rpn import *
from architecture.roi import *

from lib.database.voc_importer import *
from lib.database.util import ImageSetExpand

tf.flags.DEFINE_string("data_dir", "./data/data/full/", "Directory of data which includes \'jpg\' and \'xml\' folders.")
tf.flags.DEFINE_string("model_dir", "./data/models/", "Target directory to save the model")
tf.flags.DEFINE_string("finetune_dir", None, "Finetuned model to use. Default value lets the model start from first.")
tf.flags.DEFINE_string("model_name", "model", "Name of the model")
tf.flags.DEFINE_integer("end_step", None, "Total step to run. Default value lets the model run forever.")
tf.flags.DEFINE_integer("save_step", 100000, "Steps to save the model")
tf.flags.DEFINE_string("gpu_id", "0", "ID of gpu to use, which can be multiple.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

data_dir = FLAGS.data_dir
model_dir = FLAGS.model_dir
model_name = FLAGS.model_name
end_step = FLAGS.end_step
save_step = FLAGS.save_step
finetune_dir = FLAGS.finetune_dir
gpu_id= FLAGS.gpu_id

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

Image = tf.placeholder(tf.float32, [None, None, None, 3], name='image')
ImageInfo = tf.placeholder(tf.float32, [None, 3], name='image_info')
GroundTruth = tf.placeholder(tf.float32, [None, 5], name='ground_truth')
ConfigKey = tf.placeholder(tf.string, name='config_key')

CNN_model = 'Mobilenet'

if CNN_model == 'VGG16':
    VGG16_Builder = build.Builder(vgg16)
    VGG16_LastLayer, VGG16_Layers, VGG16_Params = VGG16_Builder(Image)
    CNN_LastLayer = VGG16_LastLayer

elif CNN_model == 'InceptionV2':
    InceptionV2_Builder = build.Builder(InceptionV2)
    InceptionV2_LastLayer, InceptionV2_Layers, InceptionV2_Params = InceptionV2_Builder(Image)
    CNN_LastLayer = InceptionV2_LastLayer

elif CNN_model == 'InceptionV4':
    InceptionV4_Builder = build.Builder(InceptionV4)
    InceptionV4_LastLayer, InceptionV4_Layers, InceptionV4_Params = InceptionV4_Builder(Image)
    CNN_LastLayer = InceptionV4_LastLayer

elif CNN_model == 'Mobilenet':
    Mobilenet_Builder = build.Builder(mobilenet)
    Mobilenet_LastLayer, Mobilenet_Layers, Mobilenet_Params = Mobilenet_Builder(Image)
    CNN_LastLayer = Mobilenet_LastLayer

elif CNN_model == 'Resnet34':
    Resnet34_Builder = build.Builder(resnet34)
    Resnet34_LastLayer, Resnet34_Layers, Resnet34_Params = Resnet34_Builder(Image)
    CNN_LastLayer = Resnet34_LastLayer

elif CNN_model == 'Resnet50':
    Resnet50_Builder = build.Builder(resnet50)
    Resnet50_LastLayer, Resnet50_Layers, Resnet50_Params = Resnet50_Builder(Image)
    CNN_LastLayer = Resnet50_LastLayer

elif CNN_model == 'Resnet101':
    Resnet101_Builder = build.Builder(resnet101)
    Resnet101_LastLayer, Resnet101_Layers, Resnet101_Params = Resnet101_Builder(Image)
    CNN_LastLayer = Resnet101_LastLayer

elif CNN_model == 'Resnet152':
    Resnet152_Builder = build.Builder(resnet152)
    Resnet152_LastLayer, Resnet152_Layers, Resnet152_Params = Resnet152_Builder(Image)
    CNN_LastLayer = Resnet152_LastLayer

# Train Model
RPN_Builder = build.Builder(rpn_train)
RPN_Proposal_BBoxes, RPN_Layers, RPN_Params = RPN_Builder([[ImageInfo, GroundTruth, ConfigKey, CNN_LastLayer], ['image_info', 'ground_truth', 'config_key', 'last_conv']])

ROI_Builder = build.Builder(roi_train)
Pred_BBoxes, ROI_Layers, ROI_Params = ROI_Builder([[CNN_LastLayer, RPN_Proposal_BBoxes, GroundTruth, ConfigKey], ['last_conv', 'rpn_proposal_bboxes', 'ground_truth', 'config_key']])
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
lr = tf.train.exponential_decay(CONFIG.TRAIN.LEARNING_RATE, global_step, CONFIG.TRAIN.STEPSIZE, 0.05, staircase=True)
# lr = 1e-3
train_op = tf.train.GradientDescentOptimizer(lr).minimize(final_loss, global_step=global_step)


def get_class_idx(name):
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    return class_names.index(name)+1

def run_sess(img, img_info, gts, model_dir, model_name, save_step, end_step=None):

    start_time = time.time()
    # rpn_cls_loss_v, rpn_bbox_loss_v, rcnn_cls_loss_v, rcnn_bbox_loss_v, global_step_v, _ = sess.run(
    #     [rpn_cls_loss, rpn_bbox_loss, rcnn_cls_loss, rcnn_bbox_loss, global_step, train_op],
    #     {
    #         Image: [img],
    #         ImageInfo: [img_info],
    #         GroundTruth: gts,
    #         ConfigKey: 'TRAIN',
    #     }
    # )

    global_step_v, final_loss_v, _ = sess.run(
        [global_step, final_loss, train_op],
        {
            Image: [img],
            ImageInfo: [img_info],
            GroundTruth: gts,
            ConfigKey: 'TRAIN',
        }
    )

    end_time = time.time()

    print("-" * 50)
    print("Step", global_step_v)
    print("Total loss : %.4f" % (final_loss_v))
    print("Time spent : %.4f" % (end_time - start_time))

    if global_step_v % save_step == 0:
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        saver.save(sess, model_dir+model_name+".ckpt", global_step=global_step)

    if global_step_v == end_step:
        print("-" * 50)
        print("-" * 50)
        print("Total time cost : %.0f" % (time.time() - tot_time))

        sys.exit()


if __name__ == '__main__':

    ConfigProto = tf.ConfigProto(allow_soft_placement=True)
    ConfigProto.gpu_options.allow_growth = True
    # ConfigProto.gpu_options.per_process_gpu_memory_fraction = 1.0
    sess = tf.InteractiveSession(config=ConfigProto)

    saver = tf.train.Saver()
    if not finetune_dir:
        tf.global_variables_initializer().run(session=sess)
    else:
        last_checkpoint = tf.train.latest_checkpoint(finetune_dir)
        saver.restore(sess, last_checkpoint)
        print('load model %s' % last_checkpoint)

    on_memory = False
    tot_time = time.time()

    if on_memory:
        org_image_set = next(voc_xml_parser(data_dir+'jpg/', data_dir+'xml/', on_memory=on_memory))
        image_set = ImageSetExpand(org_image_set)

        while True:
            for idx, (img, img_info, gt_boxes, gt_classes) in enumerate(zip(image_set['images'], image_set['image_shape'], image_set['boxes'], image_set['classes'])):
                gts = [np.concatenate([gt_boxes[i], [get_class_idx(gt_classes[i])]]) for i in range(len(gt_boxes))]

                run_sess(img, img_info, gts, model_dir=model_dir, model_name=model_name, save_step=save_step, end_step=end_step)

    else:
        while True:
            for idx, org_image_set in enumerate(
                    voc_xml_parser(data_dir+'jpg/', data_dir+'xml/', on_memory=on_memory)):

                image_set = ImageSetExpand(org_image_set)
                boxes_set, classes_set = image_set['boxes'], np.array(
                    [[get_class_idx(cls) for cls in classes] for classes in image_set['classes']])
                img = image_set['images'][0]
                img_info = image_set['image_shape'][0]
                gts = [[np.concatenate((box, [cls])) for box, cls in zip(boxes, classes)] for
                                             boxes, classes in zip(boxes_set, classes_set)][0]

                run_sess(img, img_info, gts, model_dir=model_dir, model_name=model_name, save_step=save_step, end_step=end_step)

    pass

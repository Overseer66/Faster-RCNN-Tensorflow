import cv2
import numpy as np
import time

from config import config as CONFIG

from DeepBuilder.util import SearchLayer

from architecture.mobilenet import *
from architecture.vgg import *
from architecture.rpn import *
from architecture.roi import *

from lib.database.voc_importer import *
from lib.database.util import ImageSetExpand
from lib.FRCNN.bbox_transform import BBoxTransformInverse

# Placeholder
Image = tf.placeholder(tf.float32, [None, None, None, 3], name='image')
ImageInfo = tf.placeholder(tf.float32, [None, 3], name='image_info')
# GroundTruth = tf.placeholder(tf.float32, [None, 5], name='ground_truth')
ConfigKey = tf.placeholder(tf.string, name='config_key')

# Models : VGG16, RPN, ROI
# VGG16_Builder = build.Builder(vgg16)
# CNN_LastLayer, VGG16_Layers, VGG16_Params = VGG16_Builder(Image)

Mobilenet_Builder = build.Builder(mobilenet)
CNN_LastLayer, Mobilenet_Layers, Mobilenet_Params = Mobilenet_Builder(Image)

RPN_Builder = build.Builder(rpn_test)
RPN_Proposal_BBoxes, RPN_Layers, RPN_Params = RPN_Builder([[ImageInfo, ConfigKey, CNN_LastLayer], ['image_info', 'config_key', 'last_conv']])

ROI_Builder = build.Builder(roi_test)
Pred_BBoxes, ROI_Layers, ROI_Params = ROI_Builder([[ImageInfo, CNN_LastLayer, RPN_Proposal_BBoxes], ['image_info', 'last_conv', 'rpn_proposal_bboxes']])
Pred_CLS_Prob = SearchLayer(ROI_Layers, 'cls_prob')


# definitions
def get_class_idx(name):
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    return class_names.index(name)+1


def get_class_name(idx):
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    return class_names[idx]


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

if __name__ == '__main__':
    ConfigProto = tf.ConfigProto(allow_soft_placement=True)
    ConfigProto.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=ConfigProto)
    
    # tf.global_variables_initializer().run(session=sess)
    saver = tf.train.Saver()
    # saver.restore(sess, 'data/pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt')

    on_memory = False
    if on_memory:
        org_image_set = next(voc_xml_parser('./data/data/test/jpg/', './data/data/test/xml/', on_memory=on_memory))
        image_set = ImageSetExpand(org_image_set)
        boxes_set, classes_set = image_set['boxes'], np.array([[get_class_idx(cls) for cls in classes] for classes in image_set['classes']])
        image_set['ground_truth'] = [[np.concatenate((box, [cls])) for box, cls in zip(boxes, classes)] for boxes, classes in zip(boxes_set, classes_set)]

        for idx, (img, img_info) in enumerate(zip(image_set['images'], image_set['image_shape'])):
            start_time = time.time()
            pred_boxes, pred_prob = sess.run(
                [Pred_BBoxes, Pred_CLS_Prob],
                {
                    Image: [img],
                    ImageInfo: [img_info],
                    ConfigKey: 'TEST',
                }
            )
            end_time = time.time()

            print('Figure %2d Recognition done. - %5.2f (s)' % (idx+1, end_time-start_time))

            img = org_image_set['images'][idx]
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
    else:
        for idx, org_image_set in enumerate(voc_xml_parser('./data/data/test/jpg/', './data/data/test/xml/', on_memory=on_memory)):
            image_set = ImageSetExpand(org_image_set)
            boxes_set, classes_set = image_set['boxes'], np.array([[get_class_idx(cls) for cls in classes] for classes in image_set['classes']])
            image_set['ground_truth'] = [[np.concatenate((box, [cls])) for box, cls in zip(boxes, classes)] for boxes, classes in zip(boxes_set, classes_set)]

            img = image_set['images']
            img_info = image_set['image_shape']

            start_time = time.time()
            pred_boxes, pred_prob = sess.run(
                [Pred_BBoxes, Pred_CLS_Prob],
                {
                    Image: img,
                    ImageInfo: img_info,
                    ConfigKey: 'TEST',
                }
            )
            end_time = time.time()

            print('Figure %2d Recognition done. - %5.2f (s)' % (idx+1, end_time-start_time))

            img = org_image_set['images'][0]
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






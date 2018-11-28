import cv2
import numpy as np
import time

from config import config as CONFIG

from DeepBuilder.util import SearchLayer
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
tf.flags.DEFINE_string("finetune_dir", None, "Finetuned model to use. Default value lets the model start from first.")
tf.flags.DEFINE_string("gpu_id", "0", "ID of gpu to use, which can be multiple.")
tf.flags.DEFINE_bool("on_memory", False, "Image importing method.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

data_dir = FLAGS.data_dir
finetune_dir = FLAGS.finetune_dir
on_memory = FLAGS.on_memory
gpu_id= FLAGS.gpu_id
data_dir = "./data/data/test/"

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

# Placeholder
Image = tf.placeholder(tf.float32, [None, None, None, 3], name='image')
ImageInfo = tf.placeholder(tf.float32, [None, 3], name='image_info')
# GroundTruth = tf.placeholder(tf.float32, [None, 5], name='ground_truth')
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
def vis_detections(im, class_name, dets, ax, thresh=0.5):
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

def vis_gts(ground_truth, ax):
    for i in range(len(ground_truth)):
        bbox = ground_truth[i][:4]
        cls = int(ground_truth[i][-1])

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='blue', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(get_class_name(cls-1)),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def get_true(cls, dets, ground_truth, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return []

    true_list = []
    for i in inds:
        bool_box = False
        bbox = dets[i, :4]
        score = dets[i, -1]
        for idx_gt in range(len(ground_truth)):
            iou = get_iou(bbox, ground_truth[idx_gt][:4])
            bool_iou = iou > thresh
            bool_cls = cls==ground_truth[idx_gt][-1]
            if bool_iou and bool_cls: bool_box = True
        true_list.append(bool_box)

    return true_list


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = (xB - xA + 1) * (yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def run_sess(idx, img, img_info, ground_truth, on_memory):

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

    print('Figure %2d Recognition done. - %5.2f (s)' % (idx + 1, end_time - start_time))

    prob_thresh = 0.5
    iou_thresh = 0.5

    if on_memory: img = org_image_set['images'][idx]
    else: img = org_image_set['images'][0]
    img = img[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    vis_gts(ground_truth, ax)

    true_list = []
    for idx_cls in range(n_classes - 1):
        idx_cls += 1
        cls_boxes = pred_boxes[:, 4 * idx_cls:4 * (idx_cls + 1)]
        cls_scores = pred_prob[:, idx_cls]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, 0.3)
        dets = dets[keep, :]
        vis_detections(img, get_class_name(idx_cls - 1), dets, ax, thresh=prob_thresh)
        true_list += get_true(idx_cls, dets, ground_truth, thresh=iou_thresh)

    print('Prediction Result :', sum(true_list),'/',len(true_list))

    return true_list


if __name__ == '__main__':
    ConfigProto = tf.ConfigProto(allow_soft_placement=True)
    ConfigProto.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=ConfigProto)

    saver = tf.train.Saver()
    if not finetune_dir:
        tf.global_variables_initializer().run(session=sess)
    else:
        last_checkpoint = tf.train.latest_checkpoint(finetune_dir)
        saver.restore(sess, last_checkpoint)
        print('load model %s' % last_checkpoint)

    mAP = []
    if on_memory:
        org_image_set = next(voc_xml_parser(data_dir+'jpg/', data_dir+'xml/', on_memory=on_memory))
        image_set = ImageSetExpand(org_image_set)
        boxes_set, classes_set = image_set['boxes'], np.array([[get_class_idx(cls) for cls in classes] for classes in image_set['classes']])
        image_set['ground_truth'] = [[np.concatenate((box, [cls])) for box, cls in zip(boxes, classes)] for boxes, classes in zip(boxes_set, classes_set)]


        for idx, (img, img_info, ground_truth) in enumerate(zip(image_set['images'], image_set['image_shape'], image_set['ground_truth'])):

            true_list = run_sess(idx, img, img_info, ground_truth, on_memory)
            mAP += true_list

    else:
        for idx, org_image_set in enumerate(voc_xml_parser(data_dir + 'jpg/', data_dir + 'xml/', on_memory=on_memory)):
            image_set = ImageSetExpand(org_image_set)
            boxes_set, classes_set = image_set['org_boxes'], np.array([[get_class_idx(cls) for cls in classes] for classes in image_set['classes']])
            image_set['ground_truth'] = [[np.concatenate((box, [cls])) for box, cls in zip(boxes, classes)] for boxes, classes in zip(boxes_set, classes_set)]

            img = image_set['images'][0]
            img_info = image_set['image_shape'][0]
            ground_truth = image_set['ground_truth'][0]

            true_list = run_sess(idx, img, img_info, ground_truth, on_memory)
            mAP += true_list

    if len(mAP) != 0:
        print('Total mAP :', sum(mAP)/len(mAP))

    plt.show()

    pass








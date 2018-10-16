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
from architecture.combined import *
from architecture.rpn import *
from architecture.roi import *

from lib.database.voc_importer import *
from lib.database.util import ImageSetExpand

tf.flags.DEFINE_string("data_dir", "./data/data/full/", "Directory of data which includes \'jpg\' and \'xml\' folders.")
tf.flags.DEFINE_string("finetune_dir", None, "Finetuned model to use. Default value lets the model start from first.")
tf.flags.DEFINE_string("gpu_id", "0", "ID of gpu to use, which can be multiple.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# data_dir = FLAGS.data_dir
data_dir = "./data/data/sample_10/"
finetune_dir = FLAGS.finetune_dir
gpu_id= FLAGS.gpu_id

os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

# Placeholder
Image = tf.placeholder(tf.float32, [None, None, None, 3], name='image')
ImageInfo = tf.placeholder(tf.float32, [None, 3], name='image_info')
# GroundTruth = tf.placeholder(tf.float32, [None, 5], name='ground_truth')
ConfigKey = tf.placeholder(tf.string, name='config_key')


# Models : VGG16, RPN, ROI

#VGG16
VGG16_Builder_1 = build.Builder(vgg16)
VGG16_LastLayer_1, VGG16_Layers_1, VGG16_Params_1 = VGG16_Builder_1(Image, scope="VGG16_1")

Mobilenet_Builder = build.Builder(mobilenet)
Mobilenet_LastLayer_1, Mobilenet_Layers_1, Mobilenet_Params_1 = Mobilenet_Builder(Image, scope="Mobilenet_1")
# Mobilenet_LastLayer_2, Mobilenet_Layers_2, Mobilenet_Params_2 = Mobilenet_Builder(Image, scope="Mobilenet_2")
# Mobilenet_LastLayer_3, Mobilenet_Layers_3, Mobilenet_Params_3 = Mobilenet_Builder(Image, scope="Mobilenet_3")
# Mobilenet_LastLayer_4, Mobilenet_Layers_4, Mobilenet_Params_4 = Mobilenet_Builder(Image, scope="Mobilenet_4")
# Mobilenet_LastLayer_5, Mobilenet_Layers_5, Mobilenet_Params_5 = Mobilenet_Builder(Image, scope="Mobilenet_5")
# Mobilenet_LastLayer_6, Mobilenet_Layers_6, Mobilenet_Params_6 = Mobilenet_Builder(Image, scope="Mobilenet_6")

Resnet34_Builder = build.Builder(resnet34)
Resnet34_LastLayer_1, Resnet34_Layers_1, Resnet34_Params_1 = Resnet34_Builder(Image, scope="Resnet34_1")

Resnet50_Builder = build.Builder(resnet50)
Resnet50_LastLayer_1, Resnet50_Layers_1, Resnet50_Params_1 = Resnet50_Builder(Image, scope="Resnet50_1")

Resnet101_Builder = build.Builder(resnet101)
Resnet101_LastLayer_1, Resnet101_Layers_1, Resnet101_Params_1 = Resnet101_Builder(Image, scope="Resnet101_1")

Resnet152_Builder = build.Builder(resnet152)
Resnet152_LastLayer_1, Resnet152_Layers_1, Resnet152_Params_1 = Resnet152_Builder(Image, scope="Resnet152_1")

InceptionV2_Builder = build.Builder(InceptionV2)
InceptionV2_LastLayer, InceptionV2_Layers, InceptionV2_Params = InceptionV2_Builder(Image, scope="InceptionV2")

InceptionV4_Builder = build.Builder(InceptionV4)
InceptionV4_LastLayer, InceptionV4_Layers, InceptionV4_Params = InceptionV4_Builder(Image, scope="InceptionV4")


#Combine
LastLayers = [InceptionV2_LastLayer, VGG16_LastLayer_1]
target_weight = tf.reduce_min([tf.shape(layer_i)[1] for layer_i in LastLayers])
target_height = tf.reduce_min([tf.shape(layer_i)[2] for layer_i in LastLayers])
target_size = (target_weight, target_height)
LastLayers = [tf.image.resize_images(layer_i, target_size) for layer_i in LastLayers]

Combined_Builder = build.Builder(Combined)
Combined_LastLayer, Combined_Layers, Combined_Params = Combined_Builder([LastLayers, ['Input_'+str(idx) for idx in range(len(LastLayers))]])

RPN_Builder = build.Builder(rpn_test)
RPN_Proposal_BBoxes, RPN_Layers, RPN_Params = RPN_Builder(
    [[ImageInfo, ConfigKey, Combined_LastLayer], ['image_info', 'config_key', 'last_conv']])

ROI_Builder = build.Builder(roi_test)
Pred_BBoxes, ROI_Layers, ROI_Params = ROI_Builder(
    [[ImageInfo, Combined_LastLayer, RPN_Proposal_BBoxes], ['image_info', 'last_conv', 'rpn_proposal_bboxes']])
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

def run_sess(idx, img, img_info, on_memory):

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

    if on_memory: img = org_image_set['images'][idx]
    else: img = org_image_set['images'][0]
    img = img[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    for idx in range(n_classes - 1):
        idx += 1
        cls_boxes = pred_boxes[:, 4 * idx:4 * (idx + 1)]
        cls_scores = pred_prob[:, idx]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, 0.3)
        dets = dets[keep, :]
        vis_detections(img, get_class_name(idx - 1), dets, ax)

if __name__ == '__main__':
    ConfigProto = tf.ConfigProto(allow_soft_placement=True)
    ConfigProto.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=ConfigProto)

    if not finetune_dir:
        tf.global_variables_initializer().run(session=sess)

    saver = tf.train.Saver()
    if finetune_dir:
        saver.restore(sess, finetune_dir)

    on_memory = False
    if on_memory:
        org_image_set = next(voc_xml_parser(data_dir+'jpg/', data_dir+'xml/', on_memory=on_memory))
        image_set = ImageSetExpand(org_image_set)
        boxes_set, classes_set = image_set['boxes'], np.array([[get_class_idx(cls) for cls in classes] for classes in image_set['classes']])
        image_set['ground_truth'] = [[np.concatenate((box, [cls])) for box, cls in zip(boxes, classes)] for boxes, classes in zip(boxes_set, classes_set)]


        for idx, (img, img_info) in enumerate(zip(image_set['images'], image_set['image_shape'])):

            run_sess(idx, img, img_info, on_memory)

    else:
        for idx, org_image_set in enumerate(voc_xml_parser(data_dir + 'jpg/', data_dir + 'xml/', on_memory=on_memory)):
            image_set = ImageSetExpand(org_image_set)
            boxes_set, classes_set = image_set['boxes'], np.array([[get_class_idx(cls) for cls in classes] for classes in image_set['classes']])
            image_set['ground_truth'] = [[np.concatenate((box, [cls])) for box, cls in zip(boxes, classes)] for boxes, classes in zip(boxes_set, classes_set)]

            img = image_set['images'][0]
            img_info = image_set['image_shape'][0]

            run_sess(idx, img, img_info, on_memory)

    plt.show()

    pass






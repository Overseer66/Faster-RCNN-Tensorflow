import cv2
import numpy as np

from config import config as CONFIG
from data_importer import *

from DeepBuilder.util import SearchLayer

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
# RPN_Builder = build.Builder(rpn_train)
# RPN_Proposal_BBoxes, RPN_Layers, RPN_Params = RPN_Builder([ImageInfo, GroundTruth, ConfigKey, VGG16_LastLayer])

# ROI_Builder = build.Builder(roi_train)
# Pred_BBoxes, ROI_Layers, ROI_Params = ROI_Builder([VGG16_LastLayer, RPN_Proposal_BBoxes, GroundTruth, ConfigKey])
# Pred_CLS_Prob = SearchLayer(ROI_Layers, 'cls_prob')


# Test Model
RPN_Builder = build.Builder(rpn_test)
RPN_Proposal_BBoxes, RPN_Layers, RPN_Params = RPN_Builder([ImageInfo, ConfigKey, VGG16_LastLayer])

ROI_Builder = build.Builder(roi_test)
Pred_BBoxes, ROI_Layers, ROI_Params = ROI_Builder([VGG16_LastLayer, RPN_Proposal_BBoxes, GroundTruth, ConfigKey])
Pred_CLS_Prob = SearchLayer(ROI_Layers, 'cls_prob')


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

    pred_bbox, pred_prob, rois = sess.run(
        [Pred_BBoxes, Pred_CLS_Prob, RPN_Proposal_BBoxes],
        {
            Image: img,
            ImageInfo: img_info,
            #GroundTruth: gt_boxes,
            ConfigKey: 'TRAIN',
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






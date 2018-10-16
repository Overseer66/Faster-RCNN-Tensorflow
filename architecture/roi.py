from config import config as CONFIG

from DeepBuilder import layer, activation, build, util
from lib.FRCNN.proposal_target_layer import proposal_target_layer, proposal_bbox_layer
from lib.roi_layer import roi_pooling
from lib.nms_wrapper import nms
from lib.util import ClipBoxes

n_classes = CONFIG.N_CLASSES


roi_test = (
    {'method': util.AppendInputs, },

    {'method': util.LayerSelector, 'kwargs': {'names': ['last_conv', 'rpn_proposal_bboxes']}},
    {'method': roi_pooling, 'kwargs': {'pooled_width': 7, 'pooled_height': 7, 'spatial_scale': 1.0/16}, 'name': 'pool_5'},
    {'method': activation.Transpose, 'kwargs': {'permutation': [0, 3, 1, 2]}},
    {'method': layer.flatten},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 1024, 'name': 'fc6'}},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 4096, 'name': 'fc7'}},

    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': n_classes, 'activation': None, 'name': 'cls_score'}},
    {'method': activation.Softmax, 'kwargs': {'name': 'cls_prob'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['fc7']}},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': n_classes*4, 'activation': None, 'name': 'bbox_pred'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['rpn_proposal_bboxes', 'bbox_pred', 'image_info']}},
    {'method': proposal_bbox_layer, 'kwargs': {'name': 'proposal_boxes'}}
)

roi_train = (
    {'method': util.AppendInputs, },

    # ROI Target Data
    {'method': util.LayerSelector, 'kwargs': {'names': ['rpn_proposal_bboxes', 'ground_truth', 'config_key']}},
    {'method': proposal_target_layer, 'kwargs': {'n_classes': n_classes, 'name': 'proposal_target_layer'}},
    {'method': util.LayerIndexer, 'kwargs': {'indices': [0]}},
    {'method': layer.reshape, 'kwargs': {'shape': [-1, 5], 'name': 'roi_bboxes'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['last_conv', 'roi_bboxes']}},
    {'method': roi_pooling, 'kwargs': {'pooled_width': 7, 'pooled_height': 7, 'spatial_scale': 1.0/16}, 'name': 'pool_5'},
    {'method': activation.Transpose, 'kwargs': {'permutation': [0, 3, 1, 2]}},
    {'method': layer.flatten},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 1024, 'name': 'fc6'}},
    {'method': activation.Dropout, 'kwargs': {'keep_prob': 0.5}},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': 4096, 'name': 'fc7'}},
    {'method': activation.Dropout, 'kwargs': {'keep_prob': 0.5, 'name': 'pooled_layer'}},


    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': n_classes, 'activation': None, 'name': 'cls_score'}},
    {'method': activation.Softmax, 'kwargs': {'name': 'cls_prob'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['pooled_layer']}},
    {'method': layer.fully_connected_layer, 'kwargs': {'output_size': n_classes*4, 'activation': None, 'name': 'bbox_pred'}},
)
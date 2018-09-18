from config import config as CONFIG

from DeepBuilder import layer, activation, build, util
from lib.FRCNN.anchor_layer import anchor_target_layer, split_score_layer, combine_score_layer
from lib.FRCNN.proposal_layer import proposal_layer



anchor_scales = CONFIG.ANCHOR_SCALES


rpn_test = (
    {'method': util.AppendInputs, },

    {'method': util.LayerSelector, 'kwargs': {'names': ['conv5_3']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'rpn_conv/3x3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*4], 'padding': 'VALID', 'activation': None, 'name': 'rpn_bbox_pred'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['rpn_conv/3x3']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*2], 'padding': 'VALID', 'activation': None, 'name': 'rpn_cls_score'}},

    {'method': split_score_layer, 'kwargs': {'shape': 2}},
    {'method': activation.Softmax},
    {'method': combine_score_layer, 'kwargs': {'shape': len(anchor_scales)*3*2, 'name': 'rpn_cls_prob'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['rpn_cls_prob', 'rpn_bbox_pred', 'image_info', 'config_key']}},
    {'method': proposal_layer, 'kwargs': {'feature_stride': [16,], 'anchor_scales': [8, 16, 32]}},
    {'method': layer.reshape, 'kwargs': {'shape': [-1, 5], 'name': 'rpn_proposal_bboxes'}},
)

rpn_train = (
    {'method': util.AppendInputs, },

    {'method': util.LayerSelector, 'kwargs': {'names': ['conv5_3']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'rpn_conv/3x3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*4], 'padding': 'VALID', 'activation': None, 'name': 'rpn_bbox_pred'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['rpn_conv/3x3']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, len(anchor_scales)*3*2], 'padding': 'VALID', 'activation': None, 'name': 'rpn_cls_score'}},

    # {'method': split_score_layer, 'kwargs': {'shape': 2}},
    {'method': split_score_layer, 'kwargs': {'shape': 2, 'name':'rpn_cls_score_reshape'}},
    {'method': activation.Softmax},
    {'method': combine_score_layer, 'kwargs': {'shape': len(anchor_scales)*3*2, 'name': 'rpn_cls_prob'}},

    # RPN Target Data
    {'method': util.LayerSelector, 'kwargs': {'names': ['rpn_cls_score', 'image_info', 'ground_truth', 'config_key']}},
    {'method': anchor_target_layer, 'kwargs': {'feature_stride': [16,], 'anchor_scales': [8, 16, 32], 'name':'anchor_target_layer'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['rpn_cls_prob', 'rpn_bbox_pred', 'image_info', 'config_key']}},
    {'method': proposal_layer, 'kwargs': {'feature_stride': [16,], 'anchor_scales': [8, 16, 32]}},
    {'method': layer.reshape, 'kwargs': {'shape': [-1, 5], 'name': 'rpn_proposal_bboxes'}},
)
from DeepBuilder import layer, util, activation
from architecture import layer_temp

Stem = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 32], 'stride_size':[1, 2, 2, 1], 'padding':'VALID', 'name': 'stem_1_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 32], 'stride_size':[1, 1, 1, 1], 'padding':'VALID', 'name': 'stem_2_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'stride_size':[1, 1, 1, 1], 'padding':'SAME', 'name': 'stem_3_conv'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_3_conv']}},
    {'method': layer.max_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'stride_size':[1, 2, 2, 1], 'padding':'VALID', 'name': 'stem_4_1_maxpool'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_3_conv']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 96], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'stem_4_2_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_4_1_maxpool', 'stem_4_2_conv']}},
    {'method': activation.Concatenate, 'kwargs': {'name':'stem_4_concat'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_4_concat']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 64], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'stem_5_1_conv_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 96], 'stride_size': [1, 1, 1, 1], 'padding': 'VALID', 'name': 'stem_5_1_conv_2'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_4_concat']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 64], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'stem_5_2_conv_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 1, -1, 64], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'stem_5_2_conv_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 7, -1, 64], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'stem_5_2_conv_3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 96], 'stride_size': [1, 1, 1, 1], 'padding': 'VALID', 'name': 'stem_5_2_conv_4'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_5_1_conv_2', 'stem_5_2_conv_4']}},
    {'method': activation.Concatenate, 'kwargs': {'name':'stem_5_concat'}},

    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_5_concat']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 192], 'stride_size': [1, 1, 1, 1], 'padding': 'VALID', 'name': 'stem_6_1_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_5_concat']}},
    {'method': layer.max_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'stem_6_2_maxpool'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_6_1_conv', 'stem_6_2_maxpool']}},
    {'method': activation.Concatenate, 'kwargs': {'name': 'stem_6_concat'}},
)


ModuleA = (
    {'method': util.AppendInputs, },
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_input']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 96], 'name': 'moduleA_1_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_input']}},
    {'method': layer_temp.avg_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'name': 'moduleA_2_1_avgpool'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 96], 'name': 'moduleA_2_2_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_input']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 64], 'name': 'moduleA_3_1_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 96], 'name': 'moduleA_3_2_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_input']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 64], 'name': 'moduleA_4_1_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 96], 'name': 'moduleA_4_2_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 96], 'name': 'moduleA_4_3_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_1_conv', 'moduleA_2_2_conv', 'moduleA_3_2_conv', 'moduleA_4_3_conv']}},
    {'method': activation.Concatenate, 'kwargs': {'name': 'ModuleA_concat'}},
)

ModuleA_reduction = (
    {'method': util.AppendInputs, },
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_reduction_input']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 384], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'moduleA_reduction_1_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_reduction_input']}},
    {'method': layer.max_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'moduleA_reduction_2_maxpool'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_reduction_input']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 192], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'moduleA_reduction_3_1_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 224], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'moduleA_reduction_3_2_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'moduleA_reduction_3_3_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_reduction_1_conv', 'moduleA_reduction_2_maxpool', 'moduleA_reduction_3_3_conv']}},
    {'method': activation.Concatenate, 'kwargs': {'name': 'ModuleA_concat'}},
)
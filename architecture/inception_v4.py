from DeepBuilder import layer, util, activation
from architecture import layer_temp

InceptionV4_Stem = [
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
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 192], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'stem_6_1_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_5_concat']}},
    {'method': layer.max_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'stem_6_2_maxpool'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['stem_6_1_conv', 'stem_6_2_maxpool']}},
    {'method': activation.Concatenate, 'kwargs': {'name': 'stem_6_concat'}},
]


InceptionV4_ModuleA = [{'method': layer_temp.identify, 'kwargs':{'name':'ModuleA_concat_0'}, }]
for idx in range(1,5):
    InceptionV4_ModuleA += [
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleA_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 96], 'name': 'moduleA_1_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleA_concat_'+str(idx-1)]}},
        {'method': layer_temp.avg_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'name': 'moduleA_2_1_avgpool_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 96], 'name': 'moduleA_2_2_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleA_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 64], 'name': 'moduleA_3_1_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 96], 'name': 'moduleA_3_2_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleA_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 64], 'name': 'moduleA_4_1_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 96], 'name': 'moduleA_4_2_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 96], 'name': 'moduleA_4_3_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_1_conv_'+str(idx), 'moduleA_2_2_conv_'+str(idx), 'moduleA_3_2_conv_'+str(idx), 'moduleA_4_3_conv_'+str(idx)]}},
        {'method': activation.Concatenate, 'kwargs': {'name': 'ModuleA_concat_'+str(idx)}}
    ]


InceptionV4_ModuleA_reduction = [
    {'method': layer_temp.identify, 'kwargs': {'name': 'moduleA_reduction_input'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_reduction_input']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 384], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'moduleA_reduction_1_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_reduction_input']}},
    {'method': layer.max_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'moduleA_reduction_2_maxpool'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_reduction_input']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 192], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'moduleA_reduction_3_1_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 224], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'moduleA_reduction_3_2_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'moduleA_reduction_3_3_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleA_reduction_1_conv', 'moduleA_reduction_2_maxpool', 'moduleA_reduction_3_3_conv']}},
    {'method': activation.Concatenate, 'kwargs': {'name': 'ModuleA_reduction_concat'}},
]

InceptionV4_ModuleB = [{'method': layer_temp.identify, 'kwargs':{'name':'ModuleB_concat_0'}, }]
for idx in range(1,8):
    InceptionV4_ModuleB += [
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleB_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 384], 'name': 'moduleB_1_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleB_concat_'+str(idx-1)]}},
        {'method': layer_temp.avg_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'name': 'moduleB_2_1_avgpool_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 128], 'name': 'moduleB_2_2_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleB_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 192], 'name': 'moduleB_3_1_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 1, -1, 224], 'name': 'moduleB_3_2_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 7, -1, 256], 'name': 'moduleB_3_3_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleB_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 192], 'name': 'moduleB_4_1_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 1, -1, 192], 'name': 'moduleB_4_2_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 7, -1, 224], 'name': 'moduleB_4_3_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 1, -1, 224], 'name': 'moduleB_4_4_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 7, -1, 256], 'name': 'moduleB_4_5_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['moduleB_1_conv_'+str(idx), 'moduleB_2_2_conv_'+str(idx), 'moduleB_3_3_conv_'+str(idx), 'moduleB_4_5_conv_'+str(idx)]}},
        {'method': activation.Concatenate, 'kwargs': {'name': 'ModuleB_concat_'+str(idx)}},
    ]

InceptionV4_ModuleB_reduction = [
    {'method': layer_temp.identify, 'kwargs': {'name': 'moduleB_reduction_input'}, },
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleB_reduction_input']}},
    {'method': layer.max_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'moduleB_reduction_1_maxpool'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleB_reduction_input']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 192], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'moduleB_reduction_2_1_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 192], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'moduleB_reduction_2_2_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleB_reduction_input']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 256], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'moduleB_reduction_3_1_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 1, -1, 256], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'moduleB_reduction_3_2_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 7, -1, 320], 'stride_size': [1, 1, 1, 1], 'padding': 'SAME', 'name': 'moduleB_reduction_3_3_conv'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 320], 'stride_size': [1, 2, 2, 1], 'padding': 'VALID', 'name': 'moduleB_reduction_3_4_conv'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['moduleB_reduction_1_maxpool', 'moduleB_reduction_2_2_conv', 'moduleB_reduction_3_4_conv']}},
    {'method': activation.Concatenate, 'kwargs': {'name': 'ModuleB_reduction_concat'}},
]


InceptionV4_ModuleC = [{'method': layer_temp.identify, 'kwargs':{'name':'ModuleC_concat_0'}, }]
for idx in range(1,4):
    InceptionV4_ModuleC += [
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleC_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 256], 'name': 'moduleC_1_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleC_concat_'+str(idx-1)]}},
        {'method': layer_temp.avg_pool, 'kwargs': {'kernel_size': [1, 3, 3, 1], 'name': 'moduleC_2_1_avgpool_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 256], 'name': 'moduleC_2_2_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleC_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 384], 'name': 'moduleC_3_1_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['moduleC_3_1_conv_'+str(idx)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 1, -1, 256], 'name': 'moduleC_3_2_1_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['moduleC_3_1_conv_'+str(idx)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 3, -1, 256], 'name': 'moduleC_3_2_2_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleC_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 384], 'name': 'moduleC_4_1_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 1, -1, 448], 'name': 'moduleC_4_2_conv_'+str(idx)}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 3, -1, 512], 'name': 'moduleC_4_3_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleC_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 1, -1, 256], 'name': 'moduleC_4_4_1_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['ModuleC_concat_'+str(idx-1)]}},
        {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 3, -1, 256], 'name': 'moduleC_4_4_2_conv_'+str(idx)}},
        {'method': util.LayerSelector, 'kwargs': {'names': ['moduleC_1_conv_'+str(idx), 'moduleC_2_2_conv_'+str(idx), 'moduleC_3_2_1_conv_'+str(idx), 'moduleC_3_2_2_conv_'+str(idx), 'moduleC_4_4_1_conv_'+str(idx), 'moduleC_4_4_2_conv_'+str(idx)]}},
        {'method': activation.Concatenate, 'kwargs': {'name': 'ModuleC_concat_'+str(idx)}},
    ]

InceptionV4 = InceptionV4_Stem + InceptionV4_ModuleA + InceptionV4_ModuleA_reduction + InceptionV4_ModuleB + InceptionV4_ModuleB_reduction + InceptionV4_ModuleC
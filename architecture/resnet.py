    
import tensorflow as tf
from DeepBuilder import layer, activation, build, util

resnet34 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 7, -1, 64], 'name': 'conv1'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool1'}},
    {
        # Repeat : Residual Net X 3
        # Residual Net : 3x3-64 ConvLayer X 2
        # Total 6 Layers + 1 Prev Layers
        'method': layer.repeat,
        'kwargs': {
            'count': 3,
            'layer_dict': {
                'method': layer.residual,
                'kwargs': {
                    'layer_dict': {
                        'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'res_conv1'}
                    },
                    'name': 'residual1',
                },
            },
            'name': 'resblock1',
        },
    },
    # Repeat : Residual Net X 4
    # Residual Net : 3x3-128 ConvLayer X 2
    # Total : 8 Layers + 7 Prev Layers
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'stride_size': [1, 2, 2, 1], 'name': 'resnet2_conv1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': 'resnet2_conv2'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['resblock1']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 128], 'stride_size': [1, 2, 2, 1], 'name': 'resnet2_porject_shortcut'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['resnet2_conv2', 'resnet2_project_shortcut']}},
    {'method': activation.Add, 'kwargs': {'name': ['resnet2_1']}},
    {
        'method': layer.repeat,
        'kwargs': {
            'count': 3,
            'layer_dict': {
                'method': layer.residual,
                'kwargs': {
                    'layer_dict': {
                        'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': 'resnet2_convX'}
                    },
                    'name': 'residual2',
                },
            },
            'name': 'resblock2',
        },
    },
    # Repeat : Residual Net X 4
    # Residual Net : 3x3-128 ConvLayer X 2
    # Total : 8 Layers + 7 Prev Layers
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'stride_size': [1, 2, 2, 1], 'name': 'resnet3_conv1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': 'resnet3_conv2'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['resblock2']}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 128], 'stride_size': [1, 2, 2, 1], 'name': 'resnet3_porject_shortcut'}},
    {'method': util.LayerSelector, 'kwargs': {'names': ['resnet3_conv2', 'resnet3_project_shortcut']}},
    {'method': activation.Add, 'kwargs': {'name': ['resnet3_1']}},
    {
        'method': layer.repeat,
        'kwargs': {
            'count': 3,
            'layer_dict': {
                'method': layer.residual,
                'kwargs': {
                    'layer_dict': {
                        'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': 'resnet3_convX'}
                    },
                    'name': 'residual3',
                },
            },
            'name': 'resblock3',
        },
    },
)



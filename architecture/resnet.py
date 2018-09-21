
import tensorflow as tf
from DeepBuilder import layer, activation, build

resnet = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'conv1_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'conv1_2'}},
    {
        'method': layer.residual,
        'kwargs': {
            'layer_dict': {
                'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'res_conv1_1'}
            },
            'name': 'resblock1_1'
        }
    },
    {
        'method': layer.residual,
        'kwargs': {
            'layer_dict': {
                'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'res_conv1_2'}
            },
            'name': 'resblock1_2'
        }
    },
    {
        'method': layer.residual,
        'kwargs': {
            'layer_dict': {
                'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'res_conv1_3'}
            },
            'name': 'resblock1_3'
        }
    }
)



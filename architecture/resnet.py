    
import tensorflow as tf
from DeepBuilder import layer, activation, build, util


def ResnetBlockV1(repeat, depth, numbering=''):
    numbering = str(numbering)
    ResBlock = (
        {
            # Repeat : Residual X 3
            # Residual : 3x3(64) ConvLayer X 2
            # Total 6 layers
            'method': layer.repeat,
            'kwargs': {
                'count': repeat,
                'layer_dict': {
                    'method': layer.residual,
                    'kwargs': {
                        'layer_dict': {
                            'method': layer.conv_2d, 'kwargs': {'kernel_size': (3, 3, -1, depth), 'name': 'resnet%s_conv' % numbering}
                        },
                        'name': 'residual'+numbering,
                    },
                },
                'name': 'resblock'+numbering,
            },
        },
    )
    return ResBlock


def ResnetBlockV2(repeat, depth, numbering=''):
    numbering = str(numbering)
    ResBlock = ()
    for idx in range(repeat):
        ResBlock += (
            {'method': layer.conv_2d, 'kwargs': {'kernel_size': (1, 1, -1, depth), 'name': 'resblock_%d_resnet%s_conv1' % (idx, numbering)}},
            {'method': layer.conv_2d, 'kwargs': {'kernel_size': (3, 3, -1, depth), 'name': 'resblock_%d_resnet%s_conv2' % (idx, numbering)}},
            {'method': layer.conv_2d, 'kwargs': {'kernel_size': (1, 1, -1, depth*4), 'name': 'resblock_%d_resnet%s_conv3' % (idx, numbering)}},
        )
    return ResBlock


def ProjectShortcut(depth, numbering=''):
    numbering = str(numbering)
    ps = (
        {
            'method': layer.project_shortcut,
            'kwargs': {
                'layer_dict': {
                    'method': layer.conv_2d,
                    'kwargs': {
                        'kernel_size': [3, 3, -1, depth]
                    }
                },
                'depth': depth,
                'name': 'resblock%s_ps' % numbering
            },
        },
    )
    return ps


resnet34 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 7, -1, 64], 'name': 'conv1'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool1'}},
) +\
ResnetBlockV1(3, 64, numbering=1) +\
ProjectShortcut(128, numbering=2) +\
ResnetBlockV1(3, 128, numbering=2) +\
ProjectShortcut(256, numbering=3) +\
ResnetBlockV1(5, 256, numbering=3) +\
ProjectShortcut(512, numbering=4) +\
ResnetBlockV1(2, 512, numbering=4)


resnet50 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 7, -1, 64], 'name': 'conv1'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool1'}},
) +\
ResnetBlockV2(3, 64, numbering=1) +\
ProjectShortcut(128, numbering=2) +\
ResnetBlockV2(3, 128, numbering=2) +\
ProjectShortcut(256, numbering=3) +\
ResnetBlockV2(5, 256, numbering=3) +\
ProjectShortcut(512, numbering=4) +\
ResnetBlockV2(2, 512, numbering=4)


resnet101 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 7, -1, 64], 'name': 'conv1'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool1'}},
) +\
ResnetBlockV2(3, 64, numbering=1) +\
ProjectShortcut(128, numbering=2) +\
ResnetBlockV2(3, 128, numbering=2) +\
ProjectShortcut(256, numbering=3) +\
ResnetBlockV2(22, 256, numbering=3) +\
ProjectShortcut(512, numbering=4) +\
ResnetBlockV2(2, 512, numbering=4)


resnet152 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 7, -1, 64], 'name': 'conv1'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool1'}},
) +\
ResnetBlockV2(3, 64, numbering=1) +\
ProjectShortcut(128, numbering=2) +\
ResnetBlockV2(7, 128, numbering=2) +\
ProjectShortcut(256, numbering=3) +\
ResnetBlockV2(35, 256, numbering=3) +\
ProjectShortcut(512, numbering=4) +\
ResnetBlockV2(2, 512, numbering=4)

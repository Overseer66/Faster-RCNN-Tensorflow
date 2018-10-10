    
import tensorflow as tf
from DeepBuilder import layer, activation, build, util


def get_resnet_dict(repeat, depth, step=2, kernel_size=(3,3), numbering=''):
    numbering = str(numbering)
    resnet_dict = {
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
                        'method': layer.conv_2d, 'kwargs': {'kernel_size': [kernel_size[0], kernel_size[1], -1, depth], 'name': 'resnet%s_conv' % numbering}
                    },
                    'step': step,
                    'name': 'residual'+numbering,
                },
            },
            'name': 'resblock'+numbering,
        },
    }
    return resnet_dict


def get_resnet_arch(repeat, depth, step=2, kernel_size=(3,3)):
    resnet_block = (get_resnet_dict(repeat, depth, step, kernel_size))
    return resnet_block


def get_projectshortcut_dict(depth, step=2, numbering=''):
    numbering = str(numbering)
    project_shortcut_dict = {
        'method': layer.project_shortcut,
        'kwargs': {
            'layer_dict': {
                'method': layer.conv_2d,
                'kwargs': {
                    'kernel_size': [3, 3, -1, depth]
                }
            },
            'depth': depth,
            'step': step,
            'name': 'resblock%s_ps' % numbering
        },
    }
    return project_shortcut_dict


resnet50 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [7, 7, -1, 64], 'name': 'conv1'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool1'}},
    
    # 9 Layers
    get_resnet_dict(3, 64, numbering=1),

    # 12 Layers
    get_projectshortcut_dict(128, numbering=2),
    get_resnet_dict(3, 128, numbering=2),
    
    # 18 Layers
    get_projectshortcut_dict(256, numbering=3),
    get_resnet_dict(5, 256, numbering=3),

    # 9 Layers
    get_projectshortcut_dict(512, numbering=4),
    get_resnet_dict(2, 512, numbering=4),
)



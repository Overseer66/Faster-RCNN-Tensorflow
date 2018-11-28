    
import tensorflow as tf
from DeepBuilder import layer, activation, build, util


mobilenet = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 32], 'stride_size': [1, 2, 2, 1], 'batch_norm_param': {}, 'name': 'conv1'}},

    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'batch_norm_param': {}, 'name': 'dwconv1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 64], 'batch_norm_param': {}, 'name': 'conv2'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'stride_size': [1, 2, 2, 1], 'batch_norm_param': {}, 'name': 'dwconv2'}},

    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 128], 'batch_norm_param': {}, 'name': 'conv3'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'batch_norm_param': {}, 'name': 'dwconv3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 128], 'batch_norm_param': {}, 'name': 'conv4'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'stride_size': [1, 2, 2, 1], 'batch_norm_param': {}, 'name': 'dwconv4'}},

    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 256], 'batch_norm_param': {}, 'name': 'conv5'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'batch_norm_param': {}, 'name': 'dwconv5'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 256], 'batch_norm_param': {}, 'name': 'conv6'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'stride_size': [1, 2, 2, 1], 'batch_norm_param': {}, 'name': 'dwconv6'}},

    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'batch_norm_param': {}, 'name': 'conv7'}},

    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'batch_norm_param': {}, 'name': 'dwconv7'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'batch_norm_param': {}, 'name': 'conv8'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'batch_norm_param': {}, 'name': 'dwconv8'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'batch_norm_param': {}, 'name': 'conv9'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'batch_norm_param': {}, 'name': 'dwconv9'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'batch_norm_param': {}, 'name': 'conv10'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'batch_norm_param': {}, 'name': 'dwconv10'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'batch_norm_param': {}, 'name': 'conv11'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'batch_norm_param': {}, 'name': 'dwconv11'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'batch_norm_param': {}, 'name': 'conv12'}},

    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'stride_size': [1, 2, 2, 1], 'batch_norm_param': {}, 'name': 'dwconv12'}},

    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 1024], 'batch_norm_param': {}, 'name': 'conv13'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'batch_norm_param': {}, 'name': 'dwconv13'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 1024], 'batch_norm_param': {}, 'name': 'conv14'}},
)



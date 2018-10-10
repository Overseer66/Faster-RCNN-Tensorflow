    
import tensorflow as tf
from DeepBuilder import layer, activation, build, util


mobilenet = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 32], 'stride_size': [1, 2, 2, 1], 'name': 'conv1'}},

    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'name': 'dwconv1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 64], 'name': 'conv2'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'stride_size': [1, 2, 2, 1], 'name': 'dwconv2'}},

    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 128], 'name': 'conv3'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'name': 'dwconv3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 128], 'name': 'conv4'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'stride_size': [1, 2, 2, 1], 'name': 'dwconv4'}},

    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 256], 'name': 'conv5'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'name': 'dwconv5'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 256], 'name': 'conv6'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'stride_size': [1, 2, 2, 1], 'name': 'dwconv6'}},

    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'name': 'conv7'}},

    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'name': 'dwconv7'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'name': 'conv8'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'name': 'dwconv8'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'name': 'conv9'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'name': 'dwconv9'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'name': 'conv10'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'name': 'dwconv10'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'name': 'conv11'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'name': 'dwconv11'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 512], 'name': 'conv12'}},

    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'stride_size': [1, 2, 2, 1], 'name': 'dwconv12'}},

    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 1024], 'name': 'conv13'}},
    {'method': layer.depthwise_conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 1], 'name': 'dwconv13'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [1, 1, -1, 1024], 'name': 'conv14'}},
)
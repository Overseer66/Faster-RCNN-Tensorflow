
import tensorflow as tf
from DeepBuilder import layer, activation, build

vgg16 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'conv1_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': 'conv1_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': 'conv2_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': 'conv2_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': 'conv3_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': 'conv3_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': 'conv3_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv4_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv4_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv4_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': 'pool4'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv5_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv5_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': 'conv5_3'}},
)
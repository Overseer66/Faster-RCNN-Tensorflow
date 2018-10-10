
import tensorflow as tf
from architecture import layer_temp
from DeepBuilder import layer, activation, build

vgg16_1 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': '1_conv1_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': '1_conv1_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '1_pool1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': '1_conv2_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': '1_conv2_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '1_pool2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': '1_conv3_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': '1_conv3_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': '1_conv3_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '1_pool3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '1_conv4_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '1_conv4_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '1_conv4_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '1_pool4'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '1_conv5_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '1_conv5_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '1_conv5_3'}},
    # {'method': layer_temp.dropout, 'kwargs':{'keep_prob':0.5, 'noise_shape':[1,1,1,512]}, 'name':'1_dropout'}
)

vgg16_2 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': '2_conv1_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': '2_conv1_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '2_pool1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': '2_conv2_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': '2_conv2_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '2_pool2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': '2_conv3_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': '2_conv3_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': '2_conv3_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '2_pool3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '2_conv4_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '2_conv4_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '2_conv4_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '2_pool4'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '2_conv5_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '2_conv5_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '2_conv5_3'}},
    # {'method': layer_temp.dropout, 'kwargs':{'keep_prob':0.5, 'noise_shape':[1,1,1,512]}, 'name':'2_dropout'}
)

vgg16_3 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': '3_conv1_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 64], 'name': '3_conv1_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '3_pool1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': '3_conv2_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 128], 'name': '3_conv2_2'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '3_pool2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': '3_conv3_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': '3_conv3_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 256], 'name': '3_conv3_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '3_pool3'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '3_conv4_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '3_conv4_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '3_conv4_3'}},
    {'method': layer.max_pool, 'kwargs': {'padding': 'VALID', 'name': '3_pool4'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '3_conv5_1'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '3_conv5_2'}},
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 512], 'name': '3_conv5_3'}},
    # {'method': layer_temp.dropout, 'kwargs':{'keep_prob':0.5, 'noise_shape':[1,1,1,512]}, 'name':'2_dropout'}
)
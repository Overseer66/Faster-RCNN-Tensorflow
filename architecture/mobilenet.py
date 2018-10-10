    
import tensorflow as tf
from DeepBuilder import layer, activation, build, util


mobilenet_v1 = (
    {'method': layer.conv_2d, 'kwargs': {'kernel_size': [3, 3, -1, 32], 'name': 'conv1'}},
)
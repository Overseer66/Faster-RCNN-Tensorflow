import tensorflow as tf
from DeepBuilder.util import *


def split_score(input, shape, name='_SplitScore', layer_collector=None, param_collector=None):
    input_shape = tf.shape(input)
    l = tf.reshape(input, [input_shape[0], input_shape[1], input_shape[2], shape[0], shape[1]], name=name)
    safe_append(layer_collector, l)

    return l


def combine_score(input, shape, name='_CombineScore', layer_collector=None, param_collector=None):
    input_shape = tf.shape(input)
    l = tf.reshape(input, [input_shape[0], input_shape[1], input_shape[2], shape], name=name)
    safe_append(layer_collector, l)

    return l
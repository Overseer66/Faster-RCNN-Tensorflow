from DeepBuilder.util import safe_append
import tensorflow as tf

def avg_pool(
        input,
        kernel_size=[1, 2, 2, 1],
        stride_size=[1, 1, 1, 1],
        padding='SAME',
        name='AvgPooling',
        layer_collector=None
    ):
    l = tf.nn.avg_pool(input, kernel_size, stride_size, padding, name=name)
    safe_append(layer_collector, l, name)

    return l

def global_avg_pool(
        input,
        name='GlobalAvgPooling',
        layer_collector=None
    ):
    l = tf.reduce_mean(input, [1,2], keep_dims=True, name=name)
    safe_append(layer_collector, l, name)

    return l

def dropout(
        input,
        keep_prob=0.8,
        noise_shape=None,
        name='DropOut',
        layer_collector=None
    ):
    l = tf.nn.dropout(input, keep_prob=keep_prob, noise_shape=noise_shape, name=name)
    safe_append(layer_collector, l, name)

    return l

def identify(
        input,
        name='Identify',
        layer_collector=None
    ):
    l = input
    safe_append(layer_collector, l, name)
    return l
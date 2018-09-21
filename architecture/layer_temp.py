from DeepBuilder.util import safe_append
import tensorflow as tf

def avg_pool(
        input,
        kernel_size=[1, 2, 2, 1],
        stride_size=[1, 2, 2, 1],
        padding='SAME',
        name='AvgPooling',
        layer_collector=None
    ):
    l = tf.nn.avg_pool(input, kernel_size, stride_size, padding, name=name)
    safe_append(layer_collector, l, name)

    return l
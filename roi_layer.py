from DeepBuilder.util import safe_append
from roi_pooling_layer import roi_pooling_op

def roi_pooling(input, pooled_width, pooled_height, spatial_scale, name='_ROIPool', layer_collector=None):
    l = roi_pooling_op.roi_pool(input[0], input[1], pooled_height, pooled_width, spatial_scale)
    safe_append(layer_collector, l)

    return l




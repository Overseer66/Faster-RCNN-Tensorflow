from DeepBuilder import layer, util, activation
from architecture import layer_temp

Combined = (
    {'method': util.AppendInputs, },
    {'method': layer_temp.featuremap_select, 'kwargs': {'percentage':0.2, 'name':'FeaturemapSelect'}},
    {'method': activation.Concatenate, 'kwargs': {'name': 'Combined'}},
)
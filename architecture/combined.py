from DeepBuilder import layer, util, activation
from architecture import layer_temp

Combined = (
    {'method': util.AppendInputs, },
    {'method': activation.Concatenate, 'kwargs': {'name': 'Combined'}},
)
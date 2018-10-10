from DeepBuilder import layer, util, activation
from architecture import layer_temp

Combined = (
    {'method': util.AppendInputs, },
    #TODO : select all inputs w/o demanding names of them
    {'method': util.LayerSelector, 'kwargs': {'names': ['Input_1', 'Input_2', 'Input_3', 'Input_4', 'Input_5']}},
    {'method': activation.Concatenate, 'kwargs': {'name': 'Combined'}},
)
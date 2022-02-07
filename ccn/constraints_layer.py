import numpy as np
import torch
from torch import nn

from .constraints_module import ConstraintsModule
from .constraints_group import ConstraintsGroup
from .constraint import Constraint

class ConstraintsLayer(nn.Module):
    def __init__(self, strata, num_classes):
        super(ConstraintsLayer, self).__init__()
        modules = [ConstraintsModule(stratum, num_classes) for stratum in strata]
        self.module_list = nn.ModuleList(modules)
        
    def forward(self, x, goal=None):
        for module in self.module_list:
            x = module(x, goal=goal)
        return x


def test_two_layers():
    group0 = ConstraintsGroup([
        Constraint('n1 :- 0')
    ])
    group1 = ConstraintsGroup([
        Constraint('2 :- 0'),
        Constraint('n2 :- 1'),
    ])
    group = group0 + group1 

    layer = ConstraintsLayer([group0, group1], 3)
    preds = torch.rand((1000, 3))
    updated = layer(preds)
    assert group.coherent_with(updated.numpy()).all()

    
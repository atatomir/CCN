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

def test_constraints_module():
    group = ConstraintsGroup([
        Constraint('1 :- 0'),
        Constraint('2 :- n3 4'),
        Constraint('n5 :- 6 n7 8'),
        Constraint('2 :- 9 n10'),
        Constraint('n5 :- 11 n12 n13'),
    ])
    layer = ConstraintsLayer(group, 14)
    preds = torch.rand((1000, 14))
    updated = layer(preds)
    assert group.coherent_with(updated.numpy()).all()

    
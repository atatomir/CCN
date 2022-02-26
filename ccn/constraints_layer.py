import numpy as np
import pytest
import torch
from torch import nn

from .constraints_module import ConstraintsModule
from .constraints_group import ConstraintsGroup
from .constraint import Constraint
from .clauses_group import ClausesGroup 
from .clause import Clause

class ConstraintsLayer(nn.Module):
    def __init__(self, strata, num_classes):
        super(ConstraintsLayer, self).__init__()

        # ConstraintsLayer([ConstraintsGroup], int)
        modules = [ConstraintsModule(stratum, num_classes) for stratum in strata]
        self.module_list = nn.ModuleList(modules)

    @classmethod
    def from_clauses_group(cls, group, num_classes, centrality):
        return cls(group.stratify(centrality), num_classes)
        
    def forward(self, x, goal=None, iterative=True):
        for module in self.module_list:
            x = module(x, goal=goal, iterative=iterative)
        return x

def run_layer(layer, preds):
    updated = layer(preds, iterative=True)
    updated2 = layer(preds, iterative=False)
    assert torch.isclose(updated, updated2).all()
    return updated

def test_two_modules():
    group0 = ConstraintsGroup([
        Constraint('n1 :- 0')
    ])
    group1 = ConstraintsGroup([
        Constraint('2 :- 0'),
        Constraint('n2 :- 1'),
    ])
    group = group0 + group1 

    layer = ConstraintsLayer([group0, group1], 3)
    preds = torch.rand((5000, 3))
    updated = run_layer(layer, preds)
    assert group.coherent_with(updated.numpy()).all()

def _test_many_clauses(centrality, device):
    num_classes = 30
    assignment = np.array([np.random.randint(low=0, high=2, size=num_classes)])
    clauses = ClausesGroup.random(max_clauses=150, num_classes=num_classes, coherent_with=assignment)
    layer = ConstraintsLayer.from_clauses_group(clauses, num_classes=num_classes, centrality=centrality)
    preds = torch.rand((2500, num_classes))

    layer, preds = layer.to(device), preds.to(device)

    updated = run_layer(layer, preds)
    assert clauses.coherent_with(updated.cpu().numpy()).all()
    
    difs = (updated - preds).cpu()
    assert (difs == 0.).any() 
    assert (difs != 0.).any()

def _test_many_clauses_all_measures(device):
    for centrality in ClausesGroup.centrality_measures():
        _test_many_clauses(centrality, device)

def test_many_clauses_cpu():
    _test_many_clauses_all_measures('cpu')

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_many_clauses_cuda():
    _test_many_clauses_all_measures('cuda')

def test_empty():
    assignment = np.array([np.random.randint(low=0, high=2, size=10)])
    clauses = ClausesGroup.random(max_clauses=30, num_classes=10, coherent_with=assignment)
    layer = ConstraintsLayer.from_clauses_group(clauses, num_classes=10, centrality='katz')
    preds = torch.rand((0, 10))
    updated = layer(preds)
    assert updated.shape == torch.Size([0, 10])

    
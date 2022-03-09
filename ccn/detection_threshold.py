import numpy as np
import pytest
import json
import math
import torch
from torch import nn

from .constraints_module import ConstraintsModule
from .constraints_group import ConstraintsGroup
from .constraint import Constraint
from .clauses_group import ClausesGroup 
from .slicer import Slicer
from .profiler import Profiler

class DetectionThreshold:
    def __init__(self, threshold):
        self.threshold = threshold

    def cut(self, preds, mask):
        return preds[mask, 1:]

    def uncut(self, init, mask, preds):
        preds = torch.cat((init[mask, 0].reshape(-1, 1), preds), dim=1)
        index = torch.tensor(list(range(init.shape[0])), device=mask.device)
        index = index[mask]

        #init = torch.cat((init[:, 0].reshape(-1, 1), torch.zeros_like(init[:, 1:])), dim=1)
        return init.index_copy(0, index, preds)
    
    def cutter(self, preds):
        mask = preds[:, 0] > self.threshold
        return lambda preds: self.cut(preds, mask), lambda updated: self.uncut(preds, mask, updated)

def test():
    dt = DetectionThreshold(0.2)

    preds = torch.tensor([ 
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.5, 0.6, 0.7],
        [0.0, 0.8, 0.9, 0.1],
        [0.3, 0.2, 0.3, 0.4]
    ])

    cut, uncut = dt.cutter(preds)
    cut = cut(preds)
    assert (cut == preds[[1, 3], 1:]).all()

    ones = torch.ones_like(cut)
    updated = uncut(ones)
    
    expected = torch.tensor([ 
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 1.0, 1.0, 1.0],
        [0.0, 0.8, 0.9, 0.1],
        [0.3, 1.0, 1.0, 1.0]
    ])

    assert (updated == expected).all()

def test_none():
    preds = torch.zeros(100, 100)
    dt = DetectionThreshold(0.1)

    cut, uncut = dt.cutter(preds)
    cut = cut(preds)
    assert cut.shape == torch.Size([0, 99])

    other = torch.rand(0, 99)
    assert (uncut(other) == preds).all()

def test_all():
    preds = torch.ones(100, 100)
    dt = DetectionThreshold(0.1)

    cut, uncut = dt.cutter(preds)
    cut = cut(preds)
    assert cut.shape == torch.Size([100, 99])

    other = torch.rand(100, 99)
    assert (uncut(other)[:, 1:] == other).all()

# TODO: Add cuda test



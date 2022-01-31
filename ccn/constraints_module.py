import numpy as np
import torch 

from torch import nn
from .constraint import Constraint
from .constraints_group import ConstraintsGroup

class ConstraintsModule(nn.Module):
    def __init__(self, constraints_group, num_classes):
        super(ConstraintsModule, self).__init__()
        head, body = constraints_group.encoded(num_classes)
        pos_head, neg_head = head
        pos_body, neg_body = body
        
        self.pos_head = torch.from_numpy(pos_head).float()
        self.neg_head = torch.from_numpy(neg_head).float()
        self.pos_body = torch.from_numpy(pos_body).float()
        self.neg_body = torch.from_numpy(neg_body).float()
        
    def where(self, cond, opt1, opt2):
        return opt2 + cond * (opt1 - opt2)
    
    def dimensions(self, pred):
        batch, num = pred.shape[0], pred.shape[1]
        cons = self.pos_head.shape[0]
        return batch, num, cons
    
    # Get the constraints whose body (& head) is satisfied by goal
    def satisfied_body_constraints(self, goal):
        batch, num, cons = self.dimensions(goal)
        
        # batch x cons: compute matching body
        pos_matches = torch.matmul(goal, self.pos_body.t().float())
        neg_matches = torch.matmul(1 - goal, self.neg_body.t().float())
        matches = pos_matches + neg_matches
        
        # batch x cons: compute necessary matches
        necessary = self.pos_body.sum(dim=1) + self.neg_body.sum(dim=1)
        necessary = necessary.unsqueeze(0).expand(batch, cons)
        
        # batch x cons: compute satisfying constraints
        return torch.where(matches == necessary, 1, 0)
    
    # Get the constraints whose head is not satisfied by goal
    def unsatisfied_head_constraints(self, goal):
        batch, num, cons = self.dimensions(goal)
        
        # batch x cons: compute constraints with head unsatisfied
        pos_head = torch.matmul(1 - goal, self.pos_head.t())
        neg_head = torch.matmul(goal, self.neg_head.t())
        
        return pos_head + neg_head
    
    # Get the literals unsatisfied by goal
    def unsatisfied_literals_mask(self, goal):
        batch, num, cons = self.dimensions(goal)
        
        # batch x cons x num: compute (un)satisfied literals
        goal = goal.unsqueeze(1).expand(batch, cons, num)
        return 1 - goal, goal
    
    def apply(self, preds, active_constraints=None, body_mask=None):
        batch, num, cons = self.dimensions(preds)
        
        # batch x cons x num: prepare (preds x body)
        exp_preds = preds.unsqueeze(1).expand(batch, cons, num)
        pos_body = self.pos_body.unsqueeze(0).expand(batch, cons, num)
        neg_body = self.neg_body.unsqueeze(0).expand(batch, cons, num)
        
        # ignore literals from constraints
        if body_mask != None:
            pos_mask, neg_mask = body_mask
            pos_body = pos_body * pos_mask 
            neg_body = neg_body * neg_mask
        
        # batch x cons: compute body minima
        pos_body_min = torch.min(self.where(pos_body, exp_preds, 1), dim=2).values
        neg_body_min = torch.min(self.where(neg_body, 1. - exp_preds, 1), dim=2).values
        body_min = torch.minimum(pos_body_min, neg_body_min)
        
        # ignore constraints
        if active_constraints != None:
            body_min = body_min * active_constraints
        
        # batch x cons x num: prepare (body_min x head)
        body_min = body_min.unsqueeze(2).expand(batch, cons, num)
        pos_head = self.pos_head.unsqueeze(0).expand(batch, cons, num)
        neg_head = self.neg_head.unsqueeze(0).expand(batch, cons, num)
        
        # batch x num: compute head lower and upper bounds
        pos_head_max = torch.max(body_min * pos_head, dim=1).values.float()
        neg_head_max = torch.max(body_min * neg_head, dim=1).values.float()
        assert (pos_head_max <= 1 - neg_head_max).all()

        preds = torch.maximum(pos_head_max, torch.minimum(1 - neg_head_max, preds.squeeze()))
        return preds
        
    def forward(self, preds, goal = None):
        if goal == None:
            return self.apply(preds)
        
        # constraints with head satisfied (only with full body satisfied)
        active_constraints = self.satisfied_body_constraints(goal)
        preds = self.apply(preds, active_constraints=active_constraints)

        # constraints with head not satisfied (only unsatisfied body literals)
        active_constraints = self.unsatisfied_head_constraints(goal)
        body_mask = self.unsatisfied_literals_mask(goal)
        preds = self.apply(preds, active_constraints=active_constraints, body_mask=body_mask)

        return preds

def test_constraints_module():
    group = ConstraintsGroup([
        Constraint('1 :- 0'),
        Constraint('2 :- n3 4'),
        Constraint('n5 :- 6 n7 8'),
        Constraint('2 :- 9 n10'),
        Constraint('n5 :- 11 n12 n13'),
    ])
    cm = ConstraintsModule(group, 14)
    preds = torch.rand((1000, 14))
    updated = cm(preds).numpy()
    assert group.coherent_with(updated).all()
        
def test_constraints_module_positive_goal(): 
    group = ConstraintsGroup([
        Constraint('0 :- 1 n2'),
        Constraint('3 :- 4 n5'),
        Constraint('n7 :- 7 n8'),
        Constraint('n9 :- 10 n11')
    ])

    cm = ConstraintsModule(group, 12)
    preds = torch.rand((1000, 12))
    goal = torch.tensor([1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0.]).unsqueeze(0).expand(1000, 12)
    updated = cm(preds, goal=goal).numpy()
    assert (group.coherent_with(updated).all(axis=0) == [True, False, True, False]).all()

def test_constraints_module_negative_goal():
    group = ConstraintsGroup([
        Constraint('0 :- 1 n2 3 n4'),
        Constraint('n5 :- 6 n7 8 n9')
    ])
    reduced_group = ConstraintsGroup([
        Constraint('0 :- 1 n2'),
        Constraint('n5 :- 6 n7')
    ])

    cm = ConstraintsModule(group, 10)
    preds = torch.rand((1000, 10))
    goal = torch.tensor([0., 0., 1., 1., 0., 1., 0., 1., 1., 0.]).unsqueeze(0).expand(1000, 10)
    updated = cm(preds, goal=goal).numpy()
    assert reduced_group.coherent_with(updated).all()


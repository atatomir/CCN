import numpy as np
from .constraint import Constraint

class ConstraintsGroup:
    def __init__(self, arg):
        if isinstance(arg, list):
            # ConstraintGroup([Constraint])
            self.constraints = arg
        else:
            # ConstraintGroup(string)
            with open(arg, 'r') as f:
                self.constraints = [Constraint(line) for line in f]
                
    def strata(self):
        # TODO: Implement stratification
        return [self, self]
                
    def head_encoded(self, num_classes):
        pos_head = []
        neg_head = []
        
        for constraint in self.constraints:
            pos, neg = constraint.head_encoded(num_classes)
            pos_head.append(pos)
            neg_head.append(neg)
            
        return np.array(pos_head), np.array(neg_head)
    
    def body_encoded(self, num_classes):
        pos_body = []
        neg_body = []
        
        for constraint in self.constraints:
            pos, neg = constraint.body_encoded(num_classes)
            pos_body.append(pos)
            neg_body.append(neg)
            
        return np.array(pos_body), np.array(neg_body)
            
    def encoded(self, num_classes):
        head = self.head_encoded(num_classes)
        body = self.body_encoded(num_classes)
        return head, body
    
    def coherent_with(self, preds):
        coherent = [constraint.coherent_with(preds) for constraint in self.constraints]
        return np.array(coherent).transpose()
            
    def __str__(self):
        return '\n'.join([str(constraint) for constraint in self.constraints])

def test_constraints_group_str():
    cons0 = Constraint('0 :- 1 n2')
    cons1 = Constraint('n0 :- 1')
    cons2 = Constraint('1 :- n2')
    group = ConstraintsGroup([cons0, cons1, cons2])
    assert str(group) == "0 :- 1 n2\nn0 :- 1\n1 :- n2"

def test_constraints_group_from_file():
    group = ConstraintsGroup('./constraints')
    assert str(group) == "0 :- 1 n2\nn0 :- 1\n1 :- n2"

def test_constraints_group_coherent_with():
    group = ConstraintsGroup('./constraints')
    assert (group.coherent_with(np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.2, 0.3, 0.4],
        [0.8, 0.2, 0.9, 0.4]
    ])) == np.array(
        [[False,  True, False],
        [ True,  True, False],
        [ True, False,  True]])).all()

import numpy as np
from .constraint import Constraint

class ConstraintsGroup:
    def __init__(self, arg):
        if isinstance(arg, str):
            # ConstraintGroup(string)
            with open(arg, 'r') as f:
                self.constraints = [Constraint(line) for line in f]
        else:
            # ConstraintGroup([Constraint])
            self.constraints = arg

        # Keep the initial order of constraints for coherent_with
        self.constraints_list = self.constraints
        self.constraints = frozenset(self.constraints_list)


    def __add__(self, other):
        return ConstraintsGroup(self.constraints.union(other.constraints))

    def __str__(self):
        return '\n'.join([str(constraint) for constraint in self.constraints])

    def __iter__(self):
        return iter(self.constraints)

    def __eq__(self, other):
        if not isinstance(other, ConstraintsGroup): return False
        return self.constraints == other.constraints

    def __len__(self):
        return len(self.constraints)
                
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
        coherent = [constraint.coherent_with(preds) for constraint in self.constraints_list]
        return np.array(coherent).transpose()
            

def test_str():
    cons0 = Constraint('0 :- 1 n2')
    cons1 = Constraint('n0 :- 1')
    cons2 = Constraint('1 :- n2')
    group = ConstraintsGroup([cons0, cons1, cons2])
    assert str(group) == "1 :- n2\n0 :- 1 n2\nn0 :- 1"

def test_from_file():
    group = ConstraintsGroup('../constraints/example')
    assert str(group) == "1 :- n2\n0 :- 1 n2\nn0 :- 1"

def test_coherent_with():
    group = ConstraintsGroup('../constraints/example')
    assert (group.coherent_with(np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.2, 0.3, 0.4],
        [0.8, 0.2, 0.9, 0.4]
    ])) == np.array(
        [[False,  True, False],
        [ True,  True, False],
        [ True, False,  True]])).all()

def test_add():
    c1 = Constraint('n0 :- 1 n2 3')
    c2 = Constraint('0 :- n1 n2 4')
    group0 = ConstraintsGroup([c1])
    group1 = ConstraintsGroup([c2])
    group = group0 + group1 
    assert group == ConstraintsGroup([c1, c2])

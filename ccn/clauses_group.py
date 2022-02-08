from matplotlib.pyplot import cla
import numpy as np
import pytest
from .literal import Literal
from .clause import Clause
from .constraint import Constraint
from .constraints_group import ConstraintsGroup


class ClausesGroup:
    def __init__(self, clauses):
        # ClausesGroup([Clause])
        self.clauses = frozenset(clauses)
        self.clauses_list = clauses

    def __len__(self):
        return len(self.clauses)

    def __eq__(self, other):
        if not isinstance(other, ClausesGroup): return False
        return self.clauses == other.clauses

    def __str__(self):
        return '\n'.join([str(clause) for clause in self.clauses])

    def __hash__(self):
        return hash(self.clauses)
    
    @classmethod 
    def random(cls, max_clauses, num_classes):
        clauses_count = np.random.randint(low=1, high=max_clauses)
        clauses = [Clause.random(num_classes) for i in range(clauses_count)]
        return cls(clauses)

    def resolution(self, atom):
        pos = Literal(atom, True)
        neg = Literal(atom, False)

        pos_clauses = {clause for clause in self.clauses if pos in clause}
        neg_clauses = {clause for clause in self.clauses if neg in clause}
        other_clauses = self.clauses.difference(pos_clauses).difference(neg_clauses)

        resolution_clauses = [c1.resolution(c2) for c1 in pos_clauses for c2 in neg_clauses]
        resolution_clauses = [clause for clause in resolution_clauses if clause != None]
        resolution_clauses = set(resolution_clauses)
        next_clauses = ClausesGroup(other_clauses.union(resolution_clauses))

        pos_constraints = [clause.fix_head(pos) for clause in pos_clauses]
        neg_constraints = [clause.fix_head(neg) for clause in neg_clauses]
        constraints = ConstraintsGroup(pos_constraints + neg_constraints)

        return constraints, next_clauses

    def stratify(self, atoms):
        groups = []
        clauses = self

        for atom in atoms:
            constraints, clauses = clauses.resolution(atom)
            if len(constraints): groups.append(constraints)

        if len(clauses):
            raise Exception("Unsatisfiable set of clauses")

        return groups[::-1]

    def coherent_with(self, preds):
        answer = [clause.coherent_with(preds) for clause in self.clauses_list]
        return np.array(answer).transpose()

def test_eq():
    c1 = Clause('1 n2 3')
    c2 = Clause('1 n2 n3 n2')
    c3 = Clause('1 3 n3')
    assert ClausesGroup([c1, c2, c3]) == ClausesGroup([c3, c2, c1, c1])
    assert ClausesGroup([c1, c2]) != ClausesGroup([c3, c2, c1, c1])

def test_resolution():
    c1 = Clause('1 2 3')
    c2 = Clause('1 n2 4')
    c3 = Clause('n1 4 n5')
    c4 = Clause('n1 2 6')
    c5 = Clause('2 n3 4')
    constraints, clauses = ClausesGroup([c1, c2, c3, c4, c5]).resolution(1) 
    print(clauses)

    assert constraints == ConstraintsGroup([
        Constraint('1 :- n2 n3'),
        Constraint('1 :- 2 n4'),
        Constraint('n1 :- n4 5'),
        Constraint('n1 :- n2 n6')
    ])

    assert clauses == ClausesGroup([
        Clause('2 3 4 n5'),
        Clause('2 3 6'),
        Clause('n2 4 n5'),
        Clause('4 2 n3')
    ])

def test_stratify():
    constraints = ClausesGroup([
        Clause('n0 n1'),
        Clause('n1 2'),
        Clause('1 n2')
    ]).stratify([2, 1, 0])
    assert len(constraints) == 2
    assert constraints[0] == ConstraintsGroup([
        Constraint('n1 :- 0')
    ])
    assert constraints[1] == ConstraintsGroup([
        Constraint('2 :- 1'),
        Constraint('n2 :- n1')
    ])

def test_coherent_with():
    clauses = ClausesGroup([ 
        Clause('0 1 n2 n3'),
        Clause('n0 1'),
        Clause('0 n1'),
        Clause('3 n3'),
        Clause('n2 n3')
    ])

    preds = np.array([ 
        [0.1, 0.2, 0.6, 0.7],
        [0.4, 0.7, 0.2, 0.3],
        [0.7, 0.2, 0.9, 0.8]
    ])

    assert (clauses.coherent_with(preds) == [ 
        [False, True, True, True, False],
        [True, True, False, True, True],
        [True, False, True, True, False]
    ]).all()

def test_empty_resolution():
    clauses = ClausesGroup([
        Clause('0 2'),
        Clause('n0 2'),
        Clause('1 n2'),
        Clause('n1 n2')
    ])

    with pytest.raises(Exception):
        clauses.stratify([0, 1, 2])


def test_random():
    clauses = ClausesGroup.random(max_clauses=30, num_classes=10)
    assert len(clauses) > 0 and len(clauses) <= 30
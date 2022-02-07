from ast import Constant
from .literal import Literal
from .clause import Clause
from .constraint import Constraint
from .constraints_group import ConstraintsGroup


class ClausesGroup:
    def __init__(self, clauses):
        # ClausesGroup([Clause])
        self.clauses = frozenset(clauses)

    def __len__(self):
        return len(self.clauses)

    def __eq__(self, other):
        return self.clauses == other.clauses

    def __str__(self):
        return '\n'.join([str(clause) for clause in self.clauses])

    def __hash__(self):
        return hash(self.clauses)

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

        pos_constraints = [Constraint(pos, clause) for clause in pos_clauses]
        neg_constraints = [Constraint(neg, clause) for clause in neg_clauses]
        constraints = ConstraintsGroup(pos_constraints + neg_constraints)

        return constraints, next_clauses

    def stratify(self, atoms):
        groups = []
        clauses = self.clauses

        for atom in atoms:
            constraints, clauses = clauses.resolution(atom)
            if len(constraints): groups.append(constraints)

        return groups

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
    constraints, clauses = ClausesGroup([c1, c2, c3, c4]).resolution(1) 
    print(constraints)
    print(clauses)

    assert constraints == ConstraintsGroup([
        Constraint('1 :- n2 4'),
        Constraint('1 :- 3 2')
    ])

#         1 :- 1 n2 4
# 1 :- 3 1 2
# n1 :- n1 n5 4
# n1 :- n1 6 2
# 3 n5 2 4
# 3 6 2
# n5 n2 4
    


  
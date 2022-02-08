from numpy import isin
from .literal import Literal
from .constraint import Constraint

class Clause:
    def __init__(self, literals):
        if isinstance(literals, str):
            # Clause(string)
            literals = [Literal(lit) for lit in literals.split(' ')]
            self.literals = frozenset(literals)
        elif isinstance(literals, list):
            # Clause([Literals])
            self.literals = frozenset(literals)
        else:
            # Clause(Constraint)
            body = [lit.neg() for lit in literals.body]
            self.literals = frozenset([literals.head] + body)

    def __len__(self):
        return len(self.literals)
    
    def __iter__(self):
        return iter(self.literals)

    def __eq__(self, other):
        if not isinstance(other, Clause): return False
        return self.literals == other.literals

    def __hash__(self):
        return hash(self.literals)

    def __str__(self):
        return ' '.join([str(literal) for literal in self.literals])

    def fix_head(self, head):
        if not head in self.literals:
            raise Exception('Head not in clause')
        body = [lit.neg() for lit in self.literals if lit != head]
        return Constraint(head, body)


    def always_true(self):
        for literal in self.literals:
            if literal.neg() in self.literals:
                return True 
        return False

    def resolution(self, other):
        for lit in self.literals:
            if lit.neg() in other.literals:
                result = Clause(
                    [l for l in self.literals if l != lit] +
                    [l for l in other.literals if l != lit.neg()]
                )
                return None if result.always_true() else result
        return None

    def always_false(self):
        return len(self) == 0

def test_eq():
    assert Clause('1 n2 1 2') == Clause('2 1 n2')
    assert Clause('1 n2 3 n4') != Clause('1 n2 3 4') 

def test_str():
    assert str(Clause('1 n2 1 2')) == '1 n2 2'
    assert str(Clause([Literal('1'), Literal('n2'), Literal('1'), Literal('2')])) == '1 n2 2'

def test_always_true():
    assert not Clause('1 2 n3').always_true()
    assert Clause('1 2 n3 n1').always_true()

def test_constraint():
    assert Clause('1 2 n3').fix_head(Literal('1')) == Constraint('1 :- n2 3')
    assert Clause('1 2 n3').fix_head(Literal('1')) != Constraint('n1 :- n2 3') 
    assert Clause(Constraint('2 :- 1 n0')) == Clause('2 n1 0')
    assert Clause(Constraint('n2 :- 1 n0')) != Clause('2 n1 0')

def test_resolution():
    c1 = Clause('1 n2 3')
    c2 = Clause('2 4 n5')
    assert c1.resolution(c2) == Clause('1 3 4 n5')
    c1 = Clause('1 2 n3')
    c2 = Clause('n1 2 3')
    assert c1.resolution(c2) == None 
    c1 = Clause('1 2 n3')
    c2 = Clause('n3 n4 5 6')
    assert c1.resolution(c2) == None

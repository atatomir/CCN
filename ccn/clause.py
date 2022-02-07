from .literal import Literal

class Clause:
    def __init__(self, literals):
        if isinstance(literals, str):
            # Clause(string)
            literals = [Literal(lit) for lit in literals.split(' ')]
            self.literals = set(literals)
        else:
            # Clause([Literals])
            self.literals = set(literals)

    def always_true(self):
        for literal in self.literals:
            if literal.neg() in self.literals:
                return True 
        return False
    
    def __iter__(self):
        return iter(self.literals)

    def __eq__(self, other):
        return self.literals == other.literals

    def __str__(self):
        return ' '.join([str(literal) for literal in self.literals])

def test_eq():
    assert Clause('1 n2 1 2') == Clause('2 1 n2')
    assert Clause('1 n2 3 n4') != Clause('1 n2 3 4') 

def test_str():
    assert str(Clause('1 n2 1 2')) == '1 n2 2'
    assert str(Clause([Literal('1'), Literal('n2'), Literal('1'), Literal('2')])) == '1 n2 2'

def test_always_true():
    assert not Clause('1 2 n3').always_true()
    assert Clause('1 2 n3 n1').always_true()
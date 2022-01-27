import numpy as np

class Literal:
    def __init__(self, *args):
        if len(args) == 2:
            # Literal(int, bool)
            self.atom = args[0]
            self.positive = args[1]
        else:
            # Literal(string)
            plain = args[0]
            if 'n' in plain:
                self.atom = int(plain[1:])
                self.positive = False
            else:
                self.atom = int(plain)
                self.positive = True
            
    def neg(self):
        return Literal(self.atom, not self.positive)
    
    def __str__(self):
        return str(self.atom) if self.positive else 'n' + str(self.atom)

def test_literal_init_and_str():
  assert str(Literal('13')) == "13"
  assert str(Literal('n0')) == "n0"
  assert str(Literal('0')) == "0"

def test_literal_neg():
  assert str(Literal('13').neg()) == 'n13'
  assert str(Literal('n0').neg()) == '0'
  assert str(Literal('0').neg()) == 'n0'
    
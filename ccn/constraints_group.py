import numpy as np
import networkx as nx
from torch import neg_
from .constraint import Constraint
from .literal import Literal

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

    def atoms(self):
        atoms = set()
        for constraint in self.constraints:
            atoms = atoms.union(constraint.atoms())
        return atoms

    def heads(self):
        heads = set() 
        for constraint in self.constraints:
            heads.add(constraint.head.atom)
        return heads

    def graph(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.atoms())

        for constraint in self.constraints:
            for lit in constraint.body:
                x = lit.atom 
                y = constraint.head.atom 
                G.add_edge(x, y)
                G[x][y]['body'] = lit.positive
                G[x][y]['head'] = constraint.head.positive

        return G

    def duograph(self):
        atoms = self.atoms()
        pos_atoms = [str(Literal(atom, True)) for atom in atoms]
        neg_atoms = [str(Literal(atom, False)) for atom in atoms]

        G = nx.DiGraph()
        G.add_nodes_from(pos_atoms + neg_atoms)

        for constraint in self.constraints:
            for lit in constraint.body:
                G.add_edge(str(lit), str(constraint.head))

        return G

    def stratify(self):
        G = self.graph() 

        for node in G.nodes():
            G.nodes[node]['deps'] = 0 
            G.nodes[node]['constraints'] = []

        for x, y in G.edges():
            G.nodes[y]['deps'] += 1

        for constraint in self.constraints:
            G.nodes[constraint.head.atom]['constraints'].append(constraint)

        result = []
        ready = [node for node in G.nodes() if G.nodes[node]['deps'] == 0]
        while len(ready) > 0:
            resolved = [cons for node in ready for cons in G.nodes[node]['constraints']]
            if len(resolved) > 0:
                result.append(ConstraintsGroup(resolved))
            
            next = []
            for node in ready:
                for other in G[node]:
                    G.nodes[other]['deps'] -= 1
                    if G.nodes[other]['deps'] == 0:
                        next.append(other)                    

            ready = next

        return result        

            

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

def test_atoms():
    group = ConstraintsGroup('../constraints/full')
    assert group.atoms() == set(range(41))


def test_graph():
    group = ConstraintsGroup('../constraints/example')
    graph = group.graph() 
    assert list(graph.nodes()) == [0, 1, 2]
    assert list(graph.edges()) == [(1, 0), (2, 1), (2, 0)]

def test_duograph():
    group = ConstraintsGroup('../constraints/example')
    graph = group.duograph() 
    print(graph)
    print(graph.nodes())
    print(graph.edges())
    assert list(graph.nodes()) == ['0', '1', '2', 'n0', 'n1', 'n2']
    assert list(graph.edges()) == [('1', '0'), ('1', 'n0'), ('n2', '1'), ('n2', '0'),]

def test_heads():
    group = ConstraintsGroup('../constraints/example')
    assert group.heads() == {0, 1}

def test_stratify():
    group = ConstraintsGroup([ 
        Constraint('1 :- 0'),
        Constraint('n2 :- n0 4'),
        Constraint('3 :- n1 2')
    ])
    groups = group.stratify()
    assert len(groups) == 2
    assert groups[0].heads() == {1, 2}
    assert groups[1].heads() == {3}

import torch
from torch import nn
import matplotlib.pyplot as plt

from ccn import ClausesGroup, Clause
from shapes import HalfPlane, Circle
from experiment import Experiment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ------------

H0 = HalfPlane(1, 0, -0.6)
H1 = HalfPlane(-1, 0, 0.4) 
C = Circle(0.5, 0.5, 0.20)
shapes = [H0 & -C, C, H1 & -C]

fig, ax = plt.subplots(1, len(shapes))
for i, shape in enumerate(shapes):
  shape.plot(ax[i], full=True)
plt.show()


clauses = ClausesGroup([Clause('n0 n1'), Clause('n1 n2'), Clause('0 1 2')])
constraints1 = clauses.stratify('katz')
constraints2 = clauses.stratify('rev-katz')

# -----------------------

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(2, 3),
            nn.Tanh(),
            nn.Linear(3, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        return x

model = NeuralNetwork()
print(model)
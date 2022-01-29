from matplotlib.pyplot import draw_if_interactive
import torch
import time
from torch import nn
from torch.utils.data import DataLoader 

from ccn import ConstraintsLayer, train, test, draw_classes
from shapes import ShapeDataset

class Experiment:
    def __init__(self, name, model, shapes, constraints, points = 10000):
        self.name = name
        self.model = model 
        self.shapes = shapes 
        self.constraints = constraints 

        # Build dataset
        train_data = ShapeDataset(shapes, points)
        test_data = ShapeDataset(shapes, points // 10)

        self.train_dataloader = DataLoader(train_data, batch_size = 64)
        self.test_dataloader = DataLoader(test_data, batch_size = 64)

        # Build constraints layer & optimizer
        self.clayer = ConstraintsLayer(constraints, len(shapes))
        self.loss_fn = nn.BCELoss()
        
        learning_rate = 1e-2
        self.optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.999))

    def run(self, epochs):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(self.train_dataloader, self.model, self.clayer, self.loss_fn, self.optimizer)
            self.test_loss = test(self.test_dataloader, self.model, self.clayer, self.loss_fn)
        print("Done!")
        self.draw_results()

    def experiment_path(self, dir):
        return f"{dir + self.name}-{self.test_loss:.5}-{int(time.time())}"

    def draw_results(self, path=None):
        path1 = path + '.png' if path != None else None 
        path2 = path + "-constrained.png" if path != None else None

        draw_classes(self.model, draw=(lambda ax, i: self.shapes[i].plot(ax)), path=path1)
        draw_classes(lambda x: self.clayer(self.model(x)), draw=(lambda ax, i: self.shapes[i].plot(ax)), path=path2)

    def save(self, dir='./'):
        path = self.experiment_path(dir)
        torch.save(self.model, path + ".pth")
        self.draw_results(path=path)

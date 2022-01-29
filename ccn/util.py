import torch
import pytest
import matplotlib.pyplot as plt
import numpy as np

def train(dataloader, model, clayer, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):   
        # Compute prediction error
        pred = model(X)
        constrained = clayer(pred, goal=y)
        loss = loss_fn(constrained, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

@pytest.mark.skip(reason="this is not a test")
def test(dataloader, model, clayer, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    correct = 0.
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred = clayer(pred)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.where(pred > 0.5, 1., 0.) == y).sum(dim=0)
    test_loss /= size
    correct /= size

    accuracy = ", ".join([f"{100 * rate:>0.1f}%" for rate in correct])
    print(f"Test Error: \n Accuracy: {accuracy}")
    print(f" Avg loss: {test_loss:>8f} \n")
    return test_loss

def draw_classes(model, draw = None, path=None):
    dots = np.arange(0., 1., 0.01, dtype = "float32")
    grid = torch.tensor([(x, y) for x in dots for y in dots])
    preds = model(grid).detach()

    classes = preds.shape[1]
    fig, ax = plt.subplots(1, classes)
    for i, ax in enumerate(ax):
        image = preds[:, i].view((len(dots), len(dots)))
        ax.imshow(
            image, 
            cmap='hot', 
            interpolation='nearest', 
            origin='lower', 
            extent=(0., 1., 0., 1.),
            vmin=0.,
            vmax=1.
        )
        if draw != None: draw(ax, i)    

    if path == None:
        plt.show()
    else:
        plt.savefig(path)




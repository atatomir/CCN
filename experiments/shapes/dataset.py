from torch.utils.data import Dataset
import torch

class ShapeDataset(Dataset):
    def __init__(self, shapes, count):
        super(ShapeDataset, self).__init__()

        self.values = torch.rand((count, 2,))
        self.labels = torch.tensor(
          [[1. if shape.inside(x, y) else 0. for shape in shapes] for (x, y) in self.values])

    def __len__(self):
        return len(self.values)  

    def __getitem__(self, index):
        return self.values[index], self.labels[index]
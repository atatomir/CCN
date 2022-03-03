import torch

class Watch:
    def __init__(self, callback):
        self.callback = callback

    def memory(self):
        return torch.cuda.max_memory_allocated()

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        self.prev = self.memory()

    def __exit__(self, a, b, c):
        curr = self.memory()
        self.callback((curr - self.prev) / 1024 / 1024)

class Watcher:
    def __init__(self):
        self.watches = dict() 

    def __str__(self):
        return str(self.watches)

    def register(self, name, value):
        if not name in self.watches:
            self.watches[name] = []
        self.watches[name].append(value)

    def register_for(self, name):
        return lambda value: self.register(name, value)

    def watch(self, name):
        return Watch(self.register_for(name))

    def max(self):
        result = dict()
        for key in self.watches:
            result[key] = max(self.watches[key])

        return result

    def maximum(self):
        pre = self.max()
        result = 0
        for key in pre:
            result = max(result, pre[key])

        return result

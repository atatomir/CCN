import torch
import pytest

class MaxStack:
    def __init__(self):
        self.stack = []

    def push(self, value):
        self.stack.append(value)

    def update(self, value):
        if len(self.stack):
            last = self.stack.pop()
            self.stack.append(max(last, value))

    def pop(self):
        value = self.stack.pop()
        if len(self.stack):
            last = self.stack.pop()
            self.stack.append(max(last, value))
        return value


class Watch:
    def __init__(self, stack, callback):
        self.stack = stack
        self.callback = callback

    def get_peak(self):
        return torch.cuda.max_memory_allocated()

    def reset_peak(self):
        torch.cuda.reset_peak_memory_stats()

    def __enter__(self):
        self.stack.update(self.get_peak())
        self.reset_peak()

        self.init = self.get_peak()
        self.stack.push(self.init)

    def __exit__(self, a, b, c):
        self.stack.update(self.get_peak())
        # self.reset_peak() # last block inflence moved in pop

        value = (self.stack.pop() - self.init) / 1024 / 1024
        self.callback(value)


class Profiler:
    def __init__(self):
        self.watches = dict() 
        self.stack = MaxStack()

    def register(self, name, value):
        if not name in self.watches: 
            self.watches[name] = [value]
        else: 
            self.watches[name].append(value)

    def watch(self, name):
        return Watch(self.stack, lambda value: self.register(name, value))

    def all(self):
        return self.watches

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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test():
    device = 'cuda'
    profiler = Profiler()

    for i in range(3):
        with profiler.watch('out + A + B + C'):
            out = torch.rand(512, 512, device=device)
            with profiler.watch('A'):
                a = torch.rand(1024, 1024, device=device)
            out2 = torch.rand(512, 512, device=device)
            with profiler.watch('B + C'):
                with profiler.watch('B'):
                    b = torch.rand(2048, 2048, device=device)
                with profiler.watch('C'):
                    c = torch.rand(1024, 2048, device=device)
    
    results = profiler.all()
    assert results['A'] == [4., 4., 4.]
    assert results['B'] == [16., 16., 16.]
    assert results['C'] == [8., 8., 8.]
    assert results['B + C'] == [24., 16., 16.]
    assert results['out + A + B + C'] == [30., 16., 16.]

    results = profiler.max()
    assert results['A'] == 4.
    assert results['B'] == 16. 
    assert results['C'] == 8. 
    assert results['B + C'] == 24. 
    assert results['out + A + B + C'] == 30.

    results = profiler.maximum()
    assert results == 30.

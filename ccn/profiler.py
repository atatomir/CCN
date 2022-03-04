from cProfile import Profile
import torch
import pytest

# A stack with abstract operations:
# - push & pop as usual
# - update x: set all values v to max(v, x)
class MaxStack:
    def __init__(self):
        self.stack = []

    def __len__(self):
        return len(self.stack)

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

def singleton(cls):
    def constructor(*args, **kwargs):
        if not hasattr(cls, '_instance_'):
            cls._instance_ = cls(*args, **kwargs)
        return cls._instance_ 
    return constructor

# Manages the global cuda profiling data 
@singleton
class ProfilerManager:
    def __init__(self):
        self.stack = MaxStack()

    def get_peak(self):
        return torch.cuda.max_memory_allocated()

    def reset_peak(self):
        torch.cuda.reset_peak_memory_stats()

    def enter(self):
        self.stack.update(self.get_peak())
        self.reset_peak()
        self.stack.push(0)
        return self.get_peak()

    def exit(self):
        self.stack.update(self.get_peak())
        return self.stack.pop()

# Profiler that records data from the manager
class Profiler:
    def __init__(self):
        self.watches = dict() 
        self.manager = ProfilerManager()

    @classmethod
    def shared(cls):
        if not hasattr(cls, '_shared_'):
            cls._shared_ = cls() 
        return cls._shared_

    def register(self, name, value):
        if not name in self.watches: 
            self.watches[name] = [value]
        else: 
            self.watches[name].append(value)

    def enter(self):
        return self.manager.enter()

    def exit(self, name, reference):
        value = self.manager.exit()
        value = (value - reference) / 1024 / 1024
        self.register(name, value)

    def watch(self, name):
        return Watch(name, self)

    def reset(self):
        self.watches.clear()

    def all(self):
        return self.watches

    def max(self):
        result = dict()
        for key in self.watches: result[key] = max(self.watches[key])
        return result

    def maximum(self):
        pre = self.max()
        result = 0
        for key in pre: result = max(result, pre[key])
        return result


class Watch:
    def __init__(self, name, profiler):
        self.name = name 
        self.profiler = profiler

    def __enter__(self):
        self.reference = self.profiler.enter()

    def __exit__(self, a, b, c):
        self.profiler.exit(self.name, self.reference)


def test_one_manager():
    manger = ProfilerManager()
    manager2 = ProfilerManager()
    assert id(manger) == id(manager2)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_one_profiler():
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

    profiler.reset()
    assert len(profiler.all()) == 0

def _test_nested(profiler, profiler2):
    device = 'cuda'

    with profiler.watch('test'):
        a = torch.rand(1024, 1024, device=device)
        with profiler2.watch('test2'):
            b = torch.rand(512, 512, device=device)
            del b
        with profiler2.watch('test3'):
            b = torch.rand(512 // 2, 512 // 2, device=device)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_different_profilers():
    device = 'cuda'
    profiler = Profiler()
    profiler2 = Profiler()

    _test_nested(profiler, profiler2)
    assert profiler.maximum() == 5.
    assert profiler2.maximum() == 1.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared():
    device = 'cuda'
    profiler = Profiler.shared() 
    profiler2 = Profiler.shared() 

    _test_nested(profiler, profiler2)

    assert 'test' in profiler.all()
    assert 'test2' in profiler.all()



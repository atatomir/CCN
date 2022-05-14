from midcirclebase import Experiment, NeuralNetwork, shapes, constraints2

model = NeuralNetwork()
experiment = Experiment('midcircle-rev', model, shapes, constraints2)
experiment.run(20000, device='cpu')
experiment.save(dir='./models/')
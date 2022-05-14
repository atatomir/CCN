from midcirclebase import Experiment, NeuralNetwork, shapes, constraints1

model = NeuralNetwork()
experiment = Experiment('midcircle', model, shapes, constraints1)
experiment.run(20000, device='cpu')
experiment.save(dir='./models/')
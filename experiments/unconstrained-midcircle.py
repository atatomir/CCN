from midcirclebase import Experiment, NeuralNetwork, shapes

model = NeuralNetwork()
experiment = Experiment('midcircle-unconstained', model, shapes, [])
experiment.run(20000, device='cpu')
experiment.save(dir='./models/')
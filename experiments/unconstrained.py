from base import Experiment, NeuralNetwork, shapes

model3 = NeuralNetwork()
experiment3 = Experiment('rectangles-unconstrained', model3, shapes, [])
experiment3.run(20000, 'cpu')
experiment3.save(dir='./models/')
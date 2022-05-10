from base import Experiment, NeuralNetwork, shapes, constraints1 


model1 = NeuralNetwork()
experiment = Experiment('rectangles', model1, shapes, constraints1)
experiment.run(20000, 'cpu', progressive=0)
experiment.save(dir='./models/')
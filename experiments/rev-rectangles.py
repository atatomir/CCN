from base import Experiment, NeuralNetwork, shapes, constraints2 


model2 = NeuralNetwork()
experiment2 = Experiment('rectangles-rev', model2, shapes, constraints2)
experiment2.run(20000, 'cpu')
experiment2.save(dir='./models/')
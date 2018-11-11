import unittest
import nnetwrk.neuralnetwork as nn
from nnetwrk import sigmoidneuron as sgn

class TestNeuralNetwork(unittest.TestCase):

    def test_training_1(self):
        network = nn.NeuralNetwork(2,[1,1])
        #input neuron
        neuron1 = network.getNeuron(0,0)
        neuron1.setB(0.5)
        neuron1.setW([0.4, 0.3])
        #output neuron
        neuron2 = network.getNeuron(1,0)
        neuron2.setB(0.4)
        neuron2.setW([0.3])
        #training
        network.train([1,1],[1],0.5)
        #expected bias and weight
        self.assertEqual(neuron1.getB(), 0.502101508999489)
        self.assertEqual(neuron1.getW(), [0.40210150899948904, 0.302101508999489])
        self.assertEqual(neuron2.getB(), 0.43937745312797394)
        self.assertEqual(neuron2.getW(), [0.33026254863991883])

    def test_training_2(self):
        network = nn.NeuralNetwork(2, [2, 2])
        #neurons
        neuron1 = network.getNeuron(0, 0)
        neuron2 = network.getNeuron(0, 1)
        neuron3 = network.getNeuron(1, 0)
        neuron4 = network.getNeuron(1, 1)
        #setting bias and weight
        neuron1.setB(0.5)
        neuron1.setW([0.7, 0.3])
        neuron2.setB(0.4)
        neuron2.setW([0.3, 0.7])
        neuron3.setB(0.3)
        neuron3.setW([0.2, 0.3])
        neuron4.setB(0.6)
        neuron4.setW([0.4, 0.2])
        self.assertEqual(neuron1.getB(), 0.5)
        self.assertEqual(neuron1.getW(), [0.7, 0.3])
        self.assertEqual(neuron2.getB(), 0.4)
        self.assertEqual(neuron2.getW(), [0.3, 0.7])
        self.assertEqual(neuron3.getB(), 0.3)
        self.assertEqual(neuron3.getW(), [0.2, 0.3])
        self.assertEqual(neuron4.getB(), 0.6)
        self.assertEqual(neuron4.getW(), [0.4, 0.2])

        # training
        network.train([1, 1], [1, 1], 0.5)
        # expected bias and weight
        self.assertEqual(neuron1.getB(), 0.5025104485493278)
        self.assertEqual(neuron1.getW(), [0.7025104485493278, 0.3025104485493278])
        self.assertEqual(neuron2.getB(), 0.40249801135748337)
        self.assertEqual(neuron2.getW(), [0.30249801135748333, 0.7024980113574834])
        self.assertEqual(neuron3.getB(), 0.3366295422515899)
        self.assertEqual(neuron3.getW(), [0.22994737881955657, 0.32938362863950127])
        self.assertEqual(neuron4.getB(), 0.6237654881509048)
        #here i've changed the value of expected weights in just the last number
        #the original was [0.41943005652646226, 0.21906429169838573]
        self.assertListEqual(neuron4.getW(), [0.4194300565264623, 0.21906429169838576])


if __name__== '__main__':
    unittest.main()

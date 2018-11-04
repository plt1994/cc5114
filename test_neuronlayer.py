import unittest
import nnetwrk.neuralnetwork as nn
import nnetwrk.neuronlayer
import nnetwrk.sigmoidneuron as sn


class TestNeuronLayer(unittest.TestCase):

    layer = nnetwrk.neuronlayer.NeuronLayer(2)
    layer_2 = nnetwrk.neuronlayer.NeuronLayer(4)
    sneuron = sn.SigmudNeuron(3)
    neurons = [sn.SigmudNeuron(4),sn.SigmudNeuron(3),sn.SigmudNeuron(8)]

    def test_a_init(self):
        self.assertEqual(self.layer.neurons.__len__(),2)
        for n in self.layer.neurons:
            self.assertEqual(2, n.getB())
        self.assertIsNone(self.layer.prev)
        self.assertIsNone(self.layer.next)

    def test_b_add(self):
        self.layer.addNeuron(self.sneuron)
        self.assertEqual(self.layer.neurons.__len__(), 3)
        for i, n in enumerate(self.layer.neurons):
            if i is 2:
                self.assertEqual(3, n.getB())
            else:
                self.assertEqual(2, n.getB())

    def test_c_addAll(self):
        self.layer.addAllNeuron(self.neurons)
        self.assertEqual(self.layer.getSize(), 6)

    def test_d_nextlayer(self):
        self.layer.setNext(self.layer_2)
        self.assertEqual(self.layer.next, self.layer_2)
        for n in self.layer_2.neurons:
            self.assertEqual(n.getW().__len__(),6)

    def test_e_outputs(self):
        for n in self.layer.neurons:
            self.assertEqual(n.getOutput(), 0)
        self.sneuron.setOutput(5)
        for i, n in enumerate(self.layer.neurons):
            if i is 2:
                self.assertEqual(n.getOutput(), 5)
            else:
                self.assertEqual(n.getOutput(), 0)

if __name__ == '__main__':
    unittest.main()

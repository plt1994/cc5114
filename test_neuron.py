import unittest
import iasubject.neuron as n


def f(x):
    return (x > 0) * 1


class TestNeuron(unittest.TestCase):
    pnand = n.Neuron(1,f)
    pnand.setB(3)
    pnand.setW([-2,-2])

    def test_calc_pnand(self):
        #1st test
        self.assertTrue(self.pnand.calc([0,0]))
        self.assertEqual(self.pnand.getOutput(), 1)
        self.assertTrue(self.pnand.calc([0, 1]))
        self.assertEqual(self.pnand.getOutput(), 1)
        self.assertTrue(self.pnand.calc([1, 0]))
        self.assertEqual(self.pnand.getOutput(), 1)
        self.assertFalse(self.pnand.calc([1, 1]))
        self.assertEqual(self.pnand.getOutput(), 0)

    def test_update(self):
        #2nd test
        self.assertEqual(self.pnand.w, [-2, -2])
        self.assertEqual(self.pnand.b, 3)
        self.assertEqual(self.pnand.getDelta(), 0)
        self.assertEqual(self.pnand.getOutput(), 0)
        self.pnand.setDelta(2.2)
        self.assertEqual(self.pnand.getDelta(), 2.2)


if __name__== '__main__':
    unittest.main()

import unittest
import iasubject.perceptron as p

class TestPerceptron(unittest.TestCase):
    pnand = p.Perceptron(3)
    pnand.setW([-2,-2])
    por = p.Perceptron(0)
    por.setW([1,1])
    pand = p.Perceptron(-1)
    pand.setW([1,1])


    def test_update(self):
        self.assertEqual(self.pnand.w, [-2, -2])
        self.assertEqual(self.por.w, [1,1])
        self.assertEqual(self.pand.w, [1,1])


    def test_calc_pnand(self):
        self.assertTrue(self.pnand.calc([0,0]))
        self.assertTrue(self.pnand.calc([0, 1]))
        self.assertTrue(self.pnand.calc([1, 0]))
        self.assertFalse(self.pnand.calc([1, 1]))


    def test_calc_por(self):
        self.assertFalse(self.por.calc([0,0]))
        self.assertTrue(self.por.calc([0, 1]))
        self.assertTrue(self.por.calc([1, 0]))
        self.assertTrue(self.por.calc([1, 1]))


    def test_calc_pand(self):
        self.assertFalse(self.pand.calc([0,0]))
        self.assertFalse(self.pand.calc([0, 1]))
        self.assertFalse(self.pand.calc([1, 0]))
        self.assertTrue(self.pand.calc([1, 1]))

if __name__== '__main__':
    unittest.main()


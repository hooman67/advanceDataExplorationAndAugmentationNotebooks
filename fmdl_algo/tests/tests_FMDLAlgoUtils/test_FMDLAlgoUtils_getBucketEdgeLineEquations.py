from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
from utils.FMDLAlgoUtils import getBucketEdgeLineEquations
import unittest
import numpy as np


class Test_getBucketEdgeLineEquations(unittest.TestCase):

    goundTruth_left_buck = 98
    goundTruth_bottom_buck = 417
    goundTruth_left_case = 173
    goundTruth_bottom_case = 180
    goundTruth_right_buck = 536
    goundTruth_bottom_buck = 417
    goundTruth_right_case = 467
    goundTruth_bottom_case = 180

    goundTruth_maxDiffBetweenAbsBucketEdgeSlopes = 5

    goundTruth_leftEdge_slope = -3.16
    goundTruth_leftEdge_bias = 726.6799999999997
    goundTruth_rightEdge_slope = 3.434782608695645
    goundTruth_rightEdge_bias = -1424.0434782608666



    def test_outputs_pass(self):
        resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope,\
         rightEdge_bias = getBucketEdgeLineEquations(
            (self.goundTruth_left_buck,  self.goundTruth_bottom_buck),
            (self.goundTruth_left_case,  self.goundTruth_bottom_case),
            (self.goundTruth_right_buck, self.goundTruth_bottom_buck),
            (self.goundTruth_right_case, self.goundTruth_bottom_case),
            self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes
        )


        self.assertTrue(resultAreValid)
        
        self.assertAlmostEqual(
            leftEdge_slope,
            self.goundTruth_leftEdge_slope
        )

        self.assertAlmostEqual(
            leftEdge_bias,
            self.goundTruth_leftEdge_bias
        )

        self.assertAlmostEqual(
            rightEdge_slope,
            self.goundTruth_rightEdge_slope
        )

        self.assertAlmostEqual(
            rightEdge_bias,
            self.goundTruth_rightEdge_bias
        )



    def test_outputs_lowMaxDif(self):
        resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope,\
         rightEdge_bias = getBucketEdgeLineEquations(
            (self.goundTruth_left_buck,  self.goundTruth_bottom_buck),
            (self.goundTruth_left_case,  self.goundTruth_bottom_case),
            (self.goundTruth_right_buck, self.goundTruth_bottom_buck),
            (self.goundTruth_right_case, self.goundTruth_bottom_case),
            0
        )


        self.assertFalse(resultAreValid)



    def test_outputs_badPoints1(self):
        resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope,\
         rightEdge_bias = getBucketEdgeLineEquations(
            (0,  0),
            (0,  0),
            (self.goundTruth_right_buck, self.goundTruth_bottom_buck),
            (self.goundTruth_right_case, self.goundTruth_bottom_case),
            self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes
        )

        print('\n\n\n\n\n********************************')
        print('leftEdge_slope')
        print(leftEdge_slope)
        print('leftEdge_bias')
        print(leftEdge_bias)
        print('rightEdge_slope')
        print(rightEdge_slope)
        print('rightEdge_bias')
        print(rightEdge_bias)
        print('********************************\n\n\n\n\n')

        self.assertFalse(resultAreValid)




    def test_outputs_badPoints2(self):
        resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope,\
         rightEdge_bias = getBucketEdgeLineEquations(
            ('4',  self.goundTruth_bottom_buck),
            (self.goundTruth_left_case,  self.goundTruth_bottom_case),
            (self.goundTruth_right_buck, self.goundTruth_bottom_buck),
            (self.goundTruth_right_case, self.goundTruth_bottom_case),
            self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes
        )

        print('\n\n\n\n\n********************************')
        print('leftEdge_slope')
        print(leftEdge_slope)
        print('leftEdge_bias')
        print(leftEdge_bias)
        print('rightEdge_slope')
        print(rightEdge_slope)
        print('rightEdge_bias')
        print(rightEdge_bias)
        print('********************************\n\n\n\n\n')

        self.assertFalse(resultAreValid)



    def test_outputs_badMaxDiff(self):
        resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope,\
         rightEdge_bias = getBucketEdgeLineEquations(
            (self.goundTruth_left_buck,  self.goundTruth_bottom_buck),
            (self.goundTruth_left_case,  self.goundTruth_bottom_case),
            (self.goundTruth_right_buck, self.goundTruth_bottom_buck),
            (self.goundTruth_right_case, self.goundTruth_bottom_case),
            "sdfsd"
        )

        print('\n\n\n\n\n********************************')
        print('leftEdge_slope')
        print(leftEdge_slope)
        print('leftEdge_bias')
        print(leftEdge_bias)
        print('rightEdge_slope')
        print(rightEdge_slope)
        print('rightEdge_bias')
        print(rightEdge_bias)
        print('********************************\n\n\n\n\n')

        self.assertFalse(resultAreValid)
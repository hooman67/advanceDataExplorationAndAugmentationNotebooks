from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
import cv2
from utils.FMDLAlgoUtils import areRoiBoundaryPointsIntersecting
import unittest
import numpy as np


class Test_FMDLAlgoUtils_areRoiBoundaryPointsIntersecting(unittest.TestCase):

    groundTruth_approximated_roi_boundary = np.load(
        'tests/networksAndImages-forUnitTesting/approxRoiBoundary_FMDL_2018.04.30_19.11.20.npy')

    groundTruth_approximated_roi_boundary_intersecting = np.load(
        'tests/networksAndImages-forUnitTesting/selfIntersectingROI.npy')



    def test_outputs_interesecting(self):
        data = FMDLData()
        
        data.approximated_roi_boundary = self.groundTruth_approximated_roi_boundary_intersecting

        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01,
                    "intersectingRoiMaxIterations": 10, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),
                    ('intersectingRoiMaxIterations', 0, 100, 'integer'),
                    ('minContourArea', 10, 100000, 'integer'),
                    ('roiBoundaryPointsReductionFactor',0.001, 0.1, 'float'),])
        

        areIntersecting = areRoiBoundaryPointsIntersecting(data.approximated_roi_boundary[:,0,:])

        
        self.assertTrue(areIntersecting)



    def test_outputs_notInteresecting1(self):
        data = FMDLData()
        
        data.approximated_roi_boundary = self.groundTruth_approximated_roi_boundary_intersecting

        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01,
                    "intersectingRoiMaxIterations": 10, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),
                    ('intersectingRoiMaxIterations', 0, 100, 'integer'),
                    ('minContourArea', 10, 100000, 'integer'),
                    ('roiBoundaryPointsReductionFactor',0.001, 0.1, 'float'),])
        

        areIntersecting = areRoiBoundaryPointsIntersecting(data.approximated_roi_boundary[2:,0,:])

        
        self.assertFalse(areIntersecting)



    def test_outputs_notInteresecting2(self):
        data = FMDLData()
        
        data.approximated_roi_boundary = self.groundTruth_approximated_roi_boundary

        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01,
                    "intersectingRoiMaxIterations": 10, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),
                    ('intersectingRoiMaxIterations', 0, 100, 'integer'),
                    ('minContourArea', 10, 100000, 'integer'),
                    ('roiBoundaryPointsReductionFactor',0.001, 0.1, 'float'),])
        

        areIntersecting = areRoiBoundaryPointsIntersecting(data.approximated_roi_boundary[:,0,:])

        
        self.assertFalse(areIntersecting)
        
        
        
    def test_missingRoiBoundary1(self):
        data = FMDLData()
        
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01,
                    "intersectingRoiMaxIterations": 5, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),
                    ('intersectingRoiMaxIterations', 0, 100, 'integer'),
                    ('minContourArea', 10, 100000, 'integer'),
                    ('roiBoundaryPointsReductionFactor',0.001, 0.1, 'float'),])
        
        areIntersecting = areRoiBoundaryPointsIntersecting('')
        
        self.assertTrue(areIntersecting)



    def test_missingRoiBoundary2(self):
        data = FMDLData()
        
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01,
                    "intersectingRoiMaxIterations": 5, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),
                    ('intersectingRoiMaxIterations', 0, 100, 'integer'),
                    ('minContourArea', 10, 100000, 'integer'),
                    ('roiBoundaryPointsReductionFactor',0.001, 0.1, 'float'),])
        
        areIntersecting = areRoiBoundaryPointsIntersecting([])
        
        self.assertTrue(areIntersecting)



    def test_missingRoiBoundary3(self):
        data = FMDLData()
        
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01,
                    "intersectingRoiMaxIterations": 5, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),
                    ('intersectingRoiMaxIterations', 0, 100, 'integer'),
                    ('minContourArea', 10, 100000, 'integer'),
                    ('roiBoundaryPointsReductionFactor',0.001, 0.1, 'float'),])
        
        areIntersecting = areRoiBoundaryPointsIntersecting(None)
        
        self.assertTrue(areIntersecting)
        
        
        
    def test_badRoiBoundary(self):
        data = FMDLData()
        
        data.roi_boundary_contour = np.zeros((100,1), int)
    
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01,
                    "intersectingRoiMaxIterations": 5, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),
                    ('intersectingRoiMaxIterations', 0, 100, 'integer'),
                    ('minContourArea', 10, 100000, 'integer'),
                    ('roiBoundaryPointsReductionFactor',0.001, 0.1, 'float'),])
        
        areIntersecting = areRoiBoundaryPointsIntersecting(data.roi_boundary_contour)
        
        self.assertTrue(areIntersecting)
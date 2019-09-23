from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
import cv2
from utils.FMDLAlgoUtils import getRoiBoundaryPoints
import unittest
import numpy as np


class Test_FMDLAlgoUtils_getRoiBoundaryPoints(unittest.TestCase):

    groundTruth_maxContour = np.load('tests/networksAndImages-forUnitTesting/roiBoundaryContour_FMDL_2018.04.30_19.11.20.npy')

    groundTruth_approximated_roi_boundary = np.load('tests/networksAndImages-forUnitTesting/approxRoiBoundary_FMDL_2018.04.30_19.11.20.npy')


    groundTruth_maxContour_intersecting = np.load('tests/networksAndImages-forUnitTesting/selfIntersectingROI_maxContour.npy')

    groundTruth_approximated_roi_boundary_intersecting = np.load('tests/networksAndImages-forUnitTesting/selfIntersectingROI_approxRoiBoundary.npy')



    def test_outputs(self):
        data = FMDLData()
        
        data.roi_boundary_contour = self.groundTruth_maxContour_intersecting

        print("\n\n\n\n\n***************")
        print("self.groundTruth_maxContour_intersecting")
        print(self.groundTruth_maxContour_intersecting)
        print("groundTruth_maxContour")
        print(self.groundTruth_maxContour)
        print("***************\n\n\n\n\n")
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01, "intersectingRoiMaxIterations": 10, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),('intersectingRoiMaxIterations', 0, 100, 'integer'),('minContourArea', 10, 100000, 'integer'),('roiBoundaryPointsReductionFactor',\
                     0.001, 0.1, 'float'),])
        
        validResults = getRoiBoundaryPoints(data)
        
        self.assertTrue(validResults)

        print("\nthis is the predicted approximated_roi_boundary point\n")
        print(data.approximated_roi_boundary)
        print("\nthis is the predicted groundTruth_approximated_roi_boundary point\n")
        print(self.groundTruth_approximated_roi_boundary_intersecting)
     
        self.assertTrue(np.allclose(data.approximated_roi_boundary, self.groundTruth_approximated_roi_boundary_intersecting, atol=1))


    
    def test_outputs_maxItr0_noIntersection(self):
        data = FMDLData()
        
        data.roi_boundary_contour = self.groundTruth_maxContour
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01, "intersectingRoiMaxIterations": 0, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),('intersectingRoiMaxIterations', 0, 100, 'integer'),('minContourArea', 10, 100000, 'integer'),('roiBoundaryPointsReductionFactor',\
                     0.001, 0.1, 'float'),])
        
        validResults = getRoiBoundaryPoints(data)
        
        self.assertTrue(validResults)


        print("\nthis is the predicted approximated_roi_boundary point\n")
        print(data.approximated_roi_boundary)
        print("\nthis is the predicted groundTruth_approximated_roi_boundary point\n")
        print(self.groundTruth_approximated_roi_boundary)

        self.assertTrue(np.allclose(data.approximated_roi_boundary, self.groundTruth_approximated_roi_boundary, atol=1))



    def test_outputs_noIntersection(self):
        data = FMDLData()
        
        data.roi_boundary_contour = self.groundTruth_maxContour
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01, "intersectingRoiMaxIterations": 10, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),('intersectingRoiMaxIterations', 0, 100, 'integer'),('minContourArea', 10, 100000, 'integer'),('roiBoundaryPointsReductionFactor',\
                     0.001, 0.1, 'float'),])
        
        validResults = getRoiBoundaryPoints(data)
        
        self.assertTrue(validResults)


        print("\nthis is the predicted approximated_roi_boundary point\n")
        print(data.approximated_roi_boundary)
        print("\nthis is the predicted groundTruth_approximated_roi_boundary point\n")
        print(self.groundTruth_approximated_roi_boundary)
     
        self.assertTrue(np.allclose(data.approximated_roi_boundary, self.groundTruth_approximated_roi_boundary, atol=1))


    def test_outputs_maxItr0(self):
        data = FMDLData()
        
        data.roi_boundary_contour = self.groundTruth_maxContour_intersecting

        print("\n\n\n\n\n***************")
        print("self.groundTruth_maxContour_intersecting")
        print(self.groundTruth_maxContour_intersecting)
        print("groundTruth_maxContour")
        print(self.groundTruth_maxContour)
        print("***************\n\n\n\n\n")
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01, "intersectingRoiMaxIterations": 0, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),('intersectingRoiMaxIterations', 0, 100, 'integer'),('minContourArea', 10, 100000, 'integer'),('roiBoundaryPointsReductionFactor',\
                     0.001, 0.1, 'float'),])
        
        validResults = getRoiBoundaryPoints(data)
        
        self.assertTrue(validResults)


        print("\nthis is the predicted approximated_roi_boundary point\n")
        print(data.approximated_roi_boundary)
        print("\nthis is the predicted groundTruth_approximated_roi_boundary point\n")
        print(self.groundTruth_approximated_roi_boundary_intersecting)
     
        self.assertTrue(np.allclose(data.approximated_roi_boundary, self.groundTruth_approximated_roi_boundary_intersecting, atol=1))
        
        
        
    def test_missingRoiBoundaryContour(self):
        data = FMDLData()
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01, "intersectingRoiMaxIterations": 5, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),('intersectingRoiMaxIterations', 0, 100, 'integer'),('minContourArea', 10, 100000, 'integer'),('roiBoundaryPointsReductionFactor',\
                     0.001, 0.1, 'float'),])
        
        validResults = getRoiBoundaryPoints(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.approximated_roi_boundary), 0)
        
        
        
    def test_badRoiBoundaryContour(self):
        data = FMDLData()
        
        data.roi_boundary_contour = np.zeros((100,1), int)
    
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01, "intersectingRoiMaxIterations": 5, "intersectingRoiStepSize": 0.001,},
                    [('intersectingRoiStepSize', 0.00001, 0.1, 'float'),('intersectingRoiMaxIterations', 0, 100, 'integer'),('minContourArea', 10, 100000, 'integer'),('roiBoundaryPointsReductionFactor',\
                     0.001, 0.1, 'float'),])
        
        validResults = getRoiBoundaryPoints(data.roi_boundary_contour)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.approximated_roi_boundary), 0)
        
 
        
    def test_missingConfigItems(self):
        data = FMDLData()
        
        data.roi_boundary_contour = self.groundTruth_maxContour
        
        validResults = getRoiBoundaryPoints(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.approximated_roi_boundary), 0)
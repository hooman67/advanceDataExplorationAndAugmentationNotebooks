from fmdlAlgo.FMDLAlgo import FMDLData
from utils.FMDLAlgoUtils import getApproximatedRoiBoundaryPoints
import unittest
import numpy as np


class Test_FMDLAlgoUtils_getApproximatedRoiBoundaryPoints(unittest.TestCase):
    
    groundTruth_maxContour = np.load(
        'tests/networksAndImages-forUnitTesting/roiBoundaryContour_FMDL_2018.04.30_19.11.20.npy')

    roi_boundary_points_reduction_factor = 0.01

    groundTruth_approximated_roi_boundary = np.load(
        'tests/networksAndImages-forUnitTesting/approxRoiBoundary_FMDL_2018.04.30_19.11.20.npy')


    groundTruth_maxContour_intersecting = np.load(
        'tests/networksAndImages-forUnitTesting/selfIntersectingROI_maxContour.npy')

    groundTruth_approximated_roi_boundary_intersecting = np.load('tests/networksAndImages-forUnitTesting/selfIntersectingROI_approxRoiBoundary.npy')
    
    
    def test_outputs_noIntersection(self):
        approximated_roi_boundary = getApproximatedRoiBoundaryPoints(self.groundTruth_maxContour, self.roi_boundary_points_reduction_factor)

        self.assertTrue(np.allclose(approximated_roi_boundary, self.groundTruth_approximated_roi_boundary, atol=1))
               
        
    def test_outputs_intersection(self):
        approximated_roi_boundary = getApproximatedRoiBoundaryPoints(self.groundTruth_maxContour_intersecting, self.roi_boundary_points_reduction_factor)

        self.assertTrue(np.allclose(approximated_roi_boundary, self.groundTruth_approximated_roi_boundary_intersecting, atol=1))

        
    def test_missingApproxRoi(self):
        approximated_roi_boundary = getApproximatedRoiBoundaryPoints([], self.roi_boundary_points_reduction_factor)
        
        self.assertTrue(np.allclose(approximated_roi_boundary, [], atol=1))
        
        
        
    def test_badApproxRoi1(self):
        approximated_roi_boundary = getApproximatedRoiBoundaryPoints(np.zeros(100), self.roi_boundary_points_reduction_factor)
        
        self.assertTrue(np.allclose(approximated_roi_boundary, [], atol=1))
        
        
    def test_badApproxRoi2(self):
        approximated_roi_boundary = getApproximatedRoiBoundaryPoints([1,2,3,4,5], self.roi_boundary_points_reduction_factor)
        
        self.assertTrue(np.allclose(approximated_roi_boundary, [], atol=1))
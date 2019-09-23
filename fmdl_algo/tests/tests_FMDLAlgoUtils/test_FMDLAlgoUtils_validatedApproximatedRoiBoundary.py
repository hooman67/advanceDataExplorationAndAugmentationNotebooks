###############################################################################
# Testing the FMDLAlgo.__init__ method. 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --NETWORK_LOAD_TIME_THRESHOLD
###############################################################################

from fmdlAlgo.FMDLAlgo import FMDLData
from utils.FMDLAlgoUtils import validateApproximatedRoiBoundary
import unittest
import numpy as np


class Test_FMDLAlgoUtils_validateApproximatedRoiBoundary(unittest.TestCase):
    
    approximated_roi_boundary = np.load('tests/networksAndImages-forUnitTesting/1-1-approxRoiBoundary_FMDL_2018.04.30_19.11.20.npy')
    
    
    def test_outputs(self):
        data = FMDLData()
        
        data.approximated_roi_boundary = self.approximated_roi_boundary
        
        validResults = validateApproximatedRoiBoundary(data)
        
        self.assertTrue(validResults)
        self.assertFalse(data.valid)        
        
        
        
    def test_missingApproxRoi(self):
        data = FMDLData()
        
        validResults = validateApproximatedRoiBoundary(data)
        
        self.assertFalse(validResults)
        self.assertFalse(data.valid)  
        
        
        
    def test_badApproxRoi1(self):
        data = FMDLData()
        
        data.approximated_roi_boundary = np.zeros(100)
        
        validResults = validateApproximatedRoiBoundary(data)
        
        self.assertFalse(validResults)
        self.assertFalse(data.valid)  
        
        
    def test_badApproxRoi2(self):
        data = FMDLData()
        
        data.approximated_roi_boundary = [1,2,3,4,5]
        
        validResults = validateApproximatedRoiBoundary(data)
        
        self.assertFalse(validResults)
        self.assertFalse(data.valid)  
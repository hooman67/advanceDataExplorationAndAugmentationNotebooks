###############################################################################
# Testing the FMDLAlgo.__init__ method. 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --NETWORK_LOAD_TIME_THRESHOLD
###############################################################################

from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
import cv2
from utils.FMDLAlgoUtils import postProcessRoi
import unittest
import numpy as np


class Test_FMDLAlgoUtils_postProcessRoi(unittest.TestCase):

            
    roiUncropped = cv2.imread('tests/networksAndImages-forUnitTesting/roiActualSize_FMDL_2018.04.30_19.11.20.png')
    roiUncropped = cv2.cvtColor(roiUncropped, cv2.COLOR_BGR2GRAY)
    
    
    def test_outputs1(self):
        data = FMDLData()
        
        data.roi_actualSize = self.roiUncropped
        
        testPostProcessedRoi = cv2.imread('tests/networksAndImages-forUnitTesting/closedAndEroded20_roiActualSize_FMDL_2018.04.30_19.11.20.png') 
        testPostProcessedRoi = cv2.cvtColor(testPostProcessedRoi, cv2.COLOR_BGR2GRAY)
        
        data.config = Config(
                    {
                            'closingKernelSize':                  7,
                            'closingIterations':                  20,
                            'erosionKernelSize':                  7,
                            'erosionIterations':                  20,
                    },
                    [
                            ('closingKernelSize', 1, 20),
                            ('closingIterations', 0, 20),
                            ('erosionKernelSize', 1, 20),
                            ('erosionIterations', 0, 20),
                    ])
        
        validResults = postProcessRoi(data)
        
        self.assertTrue(validResults)

        self.assertTrue(np.allclose(data.postProcessed_roi_actualSize, testPostProcessedRoi, atol=1e-04))
        
        
    def test_outputs2(self):
        data = FMDLData()
        
        data.roi_actualSize = self.roiUncropped
        
        testPostProcessedRoi = cv2.imread('tests/networksAndImages-forUnitTesting/closedAndEroded1_roiActualSize_FMDL_2018.04.30_19.11.20.png') 
        testPostProcessedRoi = cv2.cvtColor(testPostProcessedRoi, cv2.COLOR_BGR2GRAY)
        
        data.config = Config(
                    {
                            'closingKernelSize':                  7,
                            'closingIterations':                  1,
                            'erosionKernelSize':                  7,
                            'erosionIterations':                  1,
                    },
                    [
                            ('closingKernelSize', 1, 20),
                            ('closingIterations', 0, 20),
                            ('erosionKernelSize', 1, 20),
                            ('erosionIterations', 0, 20),
                    ])
        
        validResults = postProcessRoi(data)

        self.assertTrue(validResults)

        self.assertTrue(np.allclose(data.postProcessed_roi_actualSize, testPostProcessedRoi, atol=1e-04))

        
        
    def test_missingConfigItems(self):
        data = FMDLData()
        
        data.roi_actualSize = self.roiUncropped
        
        validResults = postProcessRoi(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.postProcessed_roi_actualSize), 0)
        
        
        
    def test_missingRoiActualSize(self):
        data = FMDLData()
        
        data.config = Config(
                    {
                            'closingKernelSize':                  7,
                            'closingIterations':                  1,
                            'erosionKernelSize':                  7,
                            'erosionIterations':                  1,
                    },
                    [
                            ('closingKernelSize', 1, 20),
                            ('closingIterations', 0, 20),
                            ('erosionKernelSize', 1, 20),
                            ('erosionIterations', 0, 20),
                    ])
        
        validResults = postProcessRoi(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.postProcessed_roi_actualSize), 0)
        
        

    def test_badRoiActualSize(self):
        data = FMDLData()
        
        data.roi_actualSize = np.zeros((100, 1), float)
    
        data.config = Config(
                    {
                            'closingKernelSize':                  7,
                            'closingIterations':                  1,
                            'erosionKernelSize':                  7,
                            'erosionIterations':                  1,
                    },
                    [
                            ('closingKernelSize', 1, 20),
                            ('closingIterations', 0, 20),
                            ('erosionKernelSize', 1, 20),
                            ('erosionIterations', 0, 20),
                    ])
        
        validResults = postProcessRoi(data)
        
        self.assertTrue(validResults)

        self.assertTrue(np.allclose(data.postProcessed_roi_actualSize, data.roi_actualSize, atol=1e-04))


         
    def test_failedErosionAndClosing(self):
        data = FMDLData()
        
        data.roi_actualSize = self.roiUncropped

        data.config = Config(
                    {
                            'closingKernelSize':                  -7,
                            'closingIterations':                  -20,
                            'erosionKernelSize':                  -7,
                            'erosionIterations':                  -20,
                    },
                    [
                            ('closingKernelSize', -30, 20),
                            ('closingIterations', -30, 20),
                            ('erosionKernelSize', -30, 20),
                            ('erosionIterations', -30, 20),
                    ])

        
        validResults = postProcessRoi(data)
        
        self.assertTrue(validResults)
        
        self.assertTrue(np.allclose(data.postProcessed_roi_actualSize, data.roi_actualSize, atol=1e-04))
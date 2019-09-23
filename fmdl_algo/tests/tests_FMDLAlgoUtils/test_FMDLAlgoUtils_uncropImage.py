###############################################################################
# Testing the FMDLAlgo.__init__ method. 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --NETWORK_LOAD_TIME_THRESHOLD
###############################################################################

from fmdlAlgo.FMDLAlgo import FMDLData
import cv2
from utils.FMDLAlgoUtils import uncropImage
import unittest
import numpy as np


class Test_FMDLAlgoUtils_uncropImage(unittest.TestCase):
    
    testFilePath = 'tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png'
    with open(testFilePath, "rb") as imageFile:
            f = imageFile.read()
            byteArrayImage = bytearray(f)
            
    test_roiDetection = cv2.imread('tests/networksAndImages-forUnitTesting/roidDelineatorSize_FMDL_2018.04.30_19.11.20.png')
    test_roiDetection = cv2.cvtColor(test_roiDetection, cv2.COLOR_BGR2GRAY)
    
    
    def test_outputs(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.roi_roidDelineatorSize = self.test_roiDetection
        
        data.bestBoundary = (0.36415937542915344, 0.059263408184051514, 0.9990353584289551, 0.8910735845565796)
        
        validResults = uncropImage(data)
    
        actualRoiUncropped = cv2.imread('tests/networksAndImages-forUnitTesting/roiActualSize_FMDL_2018.04.30_19.11.20.png')
        actualRoiUncropped = cv2.cvtColor(actualRoiUncropped, cv2.COLOR_BGR2GRAY)
        
        self.assertTrue(validResults)
        
        self.assertTrue(np.allclose(data.roi_actualSize, actualRoiUncropped, atol=1e-04))
        
        
        
    def test_missingBestBoundary(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.roi_roidDelineatorSize = self.test_roiDetection
        
        validResults = uncropImage(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.roi_actualSize), 0)
        
        
    
    def test_badBestBoundary(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.roi_roidDelineatorSize = self.test_roiDetection
        
        data.bestBoundary = (1, 0, 1, 0)
        
        validResults = uncropImage(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.roi_actualSize), 0)
        
        
        
    def test_missingInputImage(self):
        data = FMDLData()
        
        data.roi_roidDelineatorSize = self.test_roiDetection
        
        data.bestBoundary = (0.36415937542915344, 0.059263408184051514, 0.9990353584289551, 0.8910735845565796)
        
        validResults = uncropImage(data)

        self.assertFalse(validResults)
        self.assertEqual(len(data.roi_actualSize), 0)
        
        
        
    def test_badInputImage(self):
        data = FMDLData()
        
        data.input_image_np_cropped = np.zeros(100)
        
        data.roi_roidDelineatorSize = self.test_roiDetection
        
        data.bestBoundary = (0.36415937542915344, 0.059263408184051514, 0.9990353584289551, 0.8910735845565796)
        
        validResults = uncropImage(data)

        self.assertFalse(validResults)
        self.assertEqual(len(data.roi_actualSize), 0)
        
        
        
    def test_missingRoidDelineatorSize(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bestBoundary = (0.36415937542915344, 0.059263408184051514, 0.9990353584289551, 0.8910735845565796)
        
        validResults = uncropImage(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.roi_actualSize), 0)
        
        
        
    def test_badRoidDelineatorSize(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.roi_roidDelineatorSize = np.zeros(100)
        
        data.bestBoundary = (0.36415937542915344, 0.059263408184051514, 0.9990353584289551, 0.8910735845565796)
        
        validResults = uncropImage(data)
        
        self.assertFalse(validResults)
        
###############################################################################
# Testing the FMDLAlgo.__init__ method. 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --NETWORK_LOAD_TIME_THRESHOLD
###############################################################################

from fmdlAlgo.FMDLAlgo import FMDLData
import cv2
from utils.FMDLAlgoUtils import cropImage
import unittest
import numpy as np


class Test_FMDLAlgoUtils_cropImage(unittest.TestCase):
    
    testFilePath = 'tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png'
    with open(testFilePath, "rb") as imageFile:
            f = imageFile.read()
            byteArrayImage = bytearray(f)
    
    
    def test_missingBestBoundary(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)

        validResults = cropImage(data)
    
        self.assertFalse(validResults)
        self.assertEqual(data.input_image_np_cropped, [])
        
        
    def test_badBestBoundary(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bestBoundary = (1, 0, 1, 0)

        validResults = cropImage(data)
    
        self.assertFalse(validResults)
        self.assertEqual(len(data.input_image_np_cropped), 0)
        
        
    def test_missingInputImage(self):
        data = FMDLData()
        
        data.bestBoundary = (0.36415937542915344, 0.059263408184051514, 0.9990353584289551,\
         0.8910735845565796)

        validResults = cropImage(data)
    
        self.assertFalse(validResults)
        self.assertEqual(data.input_image_np_cropped, [])
        
        
    def test_badInputImage1(self):
        data = FMDLData()
        
        data.load_image('tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png')
        
        data.bestBoundary = (1, 0, 1, 0)

        validResults = cropImage(data)
    
        self.assertFalse(validResults)
        self.assertEqual(data.input_image_np_cropped, [])
        
        
    def test_badInputImage2(self):
        data = FMDLData()
        
        data.input_image_np = np.zeros(100)
        
        data.bestBoundary = (1, 0, 1, 0)

        validResults = cropImage(data)
    
        self.assertFalse(validResults)
        self.assertEqual(data.input_image_np_cropped, [])
        
        
    def test_outputs(self):
        data = FMDLData()
        
        actualCopedImage = cv2.imread(
            'tests/networksAndImages-forUnitTesting/cropped_FMDL_2018.04.30_19.11.20.png')
        
        data.bestBoundary = (0.36415937542915344, 0.059263408184051514, 0.9990353584289551,\
         0.8910735845565796)
        
        data.load_image(self.byteArrayImage)

        validResults = cropImage(data)
    
        self.assertTrue(validResults)
        
        self.assertTrue(np.allclose(data.input_image_np_cropped, actualCopedImage, atol=1e-04))
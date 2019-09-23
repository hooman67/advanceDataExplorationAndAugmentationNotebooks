###############################################################################
# Testing the utils.Config class 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --INFERENCE_TIME_THRESHOLD
###############################################################################

from datetime import datetime, timedelta

#Values to Test Against
INFERENCE_TIME_THRESHOLD = timedelta(seconds=1.5)

import cv2
import numpy as np
from detectors.RoiDelineator import RoiDelineator
from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
import unittest


class Test_RoiDelineator(unittest.TestCase):
    
    roiDelineator = RoiDelineator('tests/networksAndImages-forUnitTesting/roiDelineator.h5')
    test_cropped_input_image = cv2.imread('tests/networksAndImages-forUnitTesting/cropped_FMDL_2018.04.30_19.11.20.png')
    
    
    def test_inferOnSingleImage_time(self):
        data = FMDLData()
        
        test_roiDetection = cv2.imread('tests/networksAndImages-forUnitTesting/roidDelineatorSize_FMDL_2018.04.30_19.11.20.png')
        test_roiDetection = cv2.cvtColor(test_roiDetection, cv2.COLOR_BGR2GRAY)
        
        data.input_image_np_cropped = self.test_cropped_input_image
        
        data.config = Config({'roiDelineatorScoreThreshold':0.5}, [('roiDelineatorScoreThreshold', 0, 100, 'float')])
    
        
        inferenceStartTime = datetime.now()
        
        validResult = self.roiDelineator.inferOnSingleImage(data)
        inferenceTime = datetime.now() - inferenceStartTime
    
        self.assertTrue(validResult)
        self.assertLess(inferenceTime, INFERENCE_TIME_THRESHOLD)
        
        self.assertTrue(np.allclose(data.roi_roidDelineatorSize, test_roiDetection, atol=1e-04))

        
        
    def test_inferOnSingleImage_missingInputImage(self):
        data = FMDLData()
        
        data.config = Config({'roiDelineatorScoreThreshold':0.5}, [('roiDelineatorScoreThreshold', 0, 100, 'float')])
                    
        validResult = self.roiDelineator.inferOnSingleImage(data)
    
        self.assertFalse(validResult)

        self.assertEqual(len(data.roi_roidDelineatorSize), 0)
        
        
    def test_inferOnSingleImage_badInputImage(self):
        data = FMDLData()
        
        data.config = Config({'roiDelineatorScoreThreshold':0.5}, [('roiDelineatorScoreThreshold', 0, 100, 'float')])
        
        data.input_image_np_cropped = np.zeros(100)
                    
        validResult = self.roiDelineator.inferOnSingleImage(data)
    
        self.assertFalse(validResult)

        self.assertEqual(len(data.roi_roidDelineatorSize), 0)
        
        
    def test_inferOnSingleImage_missingConfig(self):
        data = FMDLData()
        
        data.input_image_np_cropped = self.test_cropped_input_image
                    
        validResult = self.roiDelineator.inferOnSingleImage(data)
    
        self.assertFalse(validResult)

        self.assertEqual(len(data.roi_roidDelineatorSize), 0)
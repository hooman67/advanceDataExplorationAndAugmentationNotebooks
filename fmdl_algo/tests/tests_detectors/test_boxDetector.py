###############################################################################
# Testing the utils.Config class 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --INFERENCE_TIME_THRESHOLD
###############################################################################

from datetime import datetime, timedelta

#Values to Test Against
INFERENCE_TIME_THRESHOLD = timedelta(seconds=4)


from detectors.BoxDetector import BoxDetector
from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
import unittest
import numpy as np


class Test_BoxDetector(unittest.TestCase):
    
    boxDetector = BoxDetector('tests/networksAndImages-forUnitTesting/bbDetector.pb')
    testFilePath = 'tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png'
    with open(testFilePath, "rb") as imageFile:
            f = imageFile.read()
            byteArrayImage = bytearray(f)
    
    
    def test_inferOnSingleImage_time(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.config = Config(
                {'boxDetectorScoreThresholdBucket':0.5,'boxDetectorScoreThresholdMatInside': 0.5,\
                "boxDetectorScoreThresholdCase": 0.5},
                [('boxDetectorScoreThresholdBucket', 0, 100, 'float'),\
                ('boxDetectorScoreThresholdMatInside', 0, 100, 'float'),\
                ('boxDetectorScoreThresholdCase', 0, 1, 'float'),])
    
        
        inferenceStartTime = datetime.now()
        validResult = self.boxDetector.inferOnSingleImage(data)
        inferenceTime = datetime.now() - inferenceStartTime
    
        self.assertTrue(validResult)
        self.assertLess(inferenceTime, INFERENCE_TIME_THRESHOLD)

        self.assertTrue(np.allclose(data.bucketBoundary, (0.3406982123851776, 0.059263408184051514,\
         1.0, 0.8910735845565796), atol=1e-04))

        self.assertTrue(np.allclose(data.bucketScore, (0.99976808), atol=1e-04))
        
        self.assertTrue(np.allclose(data.matInsideBoundary, (0.36415937542915344,\
         0.050836026668548584, 0.9990353584289551, 0.9131188988685608), atol=1e-04))

        self.assertTrue(np.allclose(data.matInsideScore, (0.99944645), atol=1e-04))


    
    
    def test_inferOnSingleImage_missingInputImage(self):
        data = FMDLData()
        validResult = self.boxDetector.inferOnSingleImage(data)
        
        self.assertFalse(validResult)
        
        for itm1, item2 in zip(data.bucketBoundary, ()):
            self.assertAlmostEqual(itm1, item2)
            
        self.assertAlmostEqual(data.bucketScore, -1)
        
        for itm1, item2 in zip(data.matInsideBoundary, ()):
            self.assertAlmostEqual(itm1, item2)
            
        self.assertAlmostEqual(data.matInsideScore, -1)
    
    
    
    def test_inferOnSingleImage_badInputImage1(self):
        data = FMDLData()
        
        data.load_image('tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png')
        
        data.config = Config(
                {'boxDetectorScoreThresholdBucket':0.5,'boxDetectorScoreThresholdMatInside': 0.5,\
                "boxDetectorScoreThresholdCase": 0.5},
                [('boxDetectorScoreThresholdBucket', 0, 100, 'float'),\
                ('boxDetectorScoreThresholdMatInside', 0, 100, 'float'),\
                ('boxDetectorScoreThresholdCase', 0, 1, 'float'),])
    
    
        validResult = self.boxDetector.inferOnSingleImage(data)
        
        self.assertFalse(validResult)
        
        for itm1, item2 in zip(data.bucketBoundary, ()):
            self.assertAlmostEqual(itm1, item2)
            
        self.assertAlmostEqual(data.bucketScore, -1)
        
        for itm1, item2 in zip(data.matInsideBoundary, ()):
            self.assertAlmostEqual(itm1, item2)
            
        self.assertAlmostEqual(data.matInsideScore, -1)
        
        
        
    def test_inferOnSingleImage_badInputImage2(self):
        data = FMDLData()
        
        data.input_image_np = np.zeros(100)
        
        data.config = Config(
                {'boxDetectorScoreThresholdBucket':0.5,'boxDetectorScoreThresholdMatInside': 0.5,\
                "boxDetectorScoreThresholdCase": 0.5},
                [('boxDetectorScoreThresholdBucket', 0, 100, 'float'),\
                ('boxDetectorScoreThresholdMatInside', 0, 100, 'float'),\
                ('boxDetectorScoreThresholdCase', 0, 1, 'float'),])
    
    
        validResult = self.boxDetector.inferOnSingleImage(data)
        
        self.assertFalse(validResult)
        
        for itm1, item2 in zip(data.bucketBoundary, ()):
            self.assertAlmostEqual(itm1, item2)
            
        self.assertAlmostEqual(data.bucketScore, -1)
        
        for itm1, item2 in zip(data.matInsideBoundary, ()):
            self.assertAlmostEqual(itm1, item2)
            
        self.assertAlmostEqual(data.matInsideScore, -1)
    
    
    
    def test_inferOnSingleImage_missingScoreThreshold(self):
        data = FMDLData()
        validResult = self.boxDetector.inferOnSingleImage(data)
        
        self.assertFalse(validResult)

        for itm1, item2 in zip(data.bucketBoundary, ()):
            self.assertAlmostEqual(itm1, item2)
            
        self.assertAlmostEqual(data.bucketScore, -1)
        
        for itm1, item2 in zip(data.matInsideBoundary, ()):
            self.assertAlmostEqual(itm1, item2)
            
        self.assertAlmostEqual(data.matInsideScore, -1)
        
    
    
    def test_thresholdPredictions(self):
        data = FMDLData()

        data.load_image(self.byteArrayImage)
        
        data.config = Config(
                {'boxDetectorScoreThresholdBucket':0,'boxDetectorScoreThresholdMatInside': 0,\
                "boxDetectorScoreThresholdCase": 0},
                [('boxDetectorScoreThresholdBucket', 0, 100, 'float'),\
                ('boxDetectorScoreThresholdMatInside', 0, 100, 'float'),\
                ('boxDetectorScoreThresholdCase', 0, 1, 'float'),])
    
    
        validResult = self.boxDetector.inferOnSingleImage(data)
        
        self.assertTrue(validResult)
        
        self.assertTrue(np.allclose(data.bucketBoundary, (0.3406982123851776, 0.059263408184051514,\
         1.0, 0.8910735845565796), atol=1e-04))

        self.assertTrue(np.allclose(data.bucketScore, (0.99976808), atol=1e-04))

        self.assertTrue(np.allclose(data.matInsideBoundary, (0.36415937542915344,\
         0.050836026668548584, 0.9990353584289551, 0.9131188988685608), atol=1e-04))

        self.assertTrue(np.allclose(data.matInsideScore, (0.99944645), atol=1e-04))
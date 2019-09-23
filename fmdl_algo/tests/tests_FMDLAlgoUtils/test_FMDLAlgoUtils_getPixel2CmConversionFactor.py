###############################################################################
# Testing the FMDLAlgoUtils.getPixel2CmConversionFactor method. 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --None
###############################################################################

from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
from utils.FMDLAlgoUtils import getPixel2CmConversionFactor
import unittest
import numpy as np


class Test_GetPixel2CmConversionFactor(unittest.TestCase):
    
    testFilePath = 'tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png'
    with open(testFilePath, "rb") as imageFile:
            f = imageFile.read()
            byteArrayImage = bytearray(f)
    
    
    def test_missingBucketBoundary(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
    
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,'maxDiffBetweenAbsBucketEdgeSlopes': 3},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        validResults = getPixel2CmConversionFactor(data)
        
        self.assertFalse(validResults)
        self.assertEqual(data.pixel2CM_conversion_factor , -1)
        self.assertEqual(data.bucketWidthPX, -1)
        self.assertEqual(data.bucketWidthPointsXCord, [])

    
    
    def test_badBucketBoundary(self):
        data = FMDLData()
        
    
        data.config = Config(
                    {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,'maxDiffBetweenAbsBucketEdgeSlopes': 3},
                    [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
       
        
        data.load_image(self.byteArrayImage)
        
        
        data.bucketBoundary = (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0)
        validResults1 = getPixel2CmConversionFactor(data)
        
        
        data.bucketBoundary = (0, -0.8910735845565796, 0, 1.0)
        validResults2 = getPixel2CmConversionFactor(data)
        
        self.assertFalse(validResults1)
        self.assertFalse(validResults2)
        self.assertEqual(data.pixel2CM_conversion_factor , -1)
        self.assertEqual(data.bucketWidthPX, -1)
        self.assertEqual(data.bucketWidthPointsXCord, [])
    
    
    
    def test_bucketBoundaryWithWrongAspectRatio(self):
        data = FMDLData()

        
        data.load_image(self.byteArrayImage)
        
        
        data.bucketBoundary = (1,2,3,4)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,'maxDiffBetweenAbsBucketEdgeSlopes': 3},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        validResults = getPixel2CmConversionFactor(data)
        
        self.assertFalse(validResults)
        
    
    
    def test_missingInputImage(self):
        data = FMDLData()
        
        data.bucketBoundary = (1,2,3,4)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,'maxDiffBetweenAbsBucketEdgeSlopes': 3},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        validResults = getPixel2CmConversionFactor(data)
        
        self.assertFalse(validResults)
        self.assertEqual(data.pixel2CM_conversion_factor , -1)
        self.assertEqual(data.bucketWidthPX, -1)
        self.assertEqual(data.bucketWidthPointsXCord, [])

    
    
    def test_badInputImage1(self):
        data = FMDLData()
        
        data.load_image('tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png')
        
        data.bucketBoundary = (1,2,3,4)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,'maxDiffBetweenAbsBucketEdgeSlopes': 3},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        validResults = getPixel2CmConversionFactor(data)
        
        self.assertFalse(validResults)
        self.assertEqual(data.pixel2CM_conversion_factor , -1)
        self.assertEqual(data.bucketWidthPX, -1)
        self.assertEqual(data.bucketWidthPointsXCord, [])
        
        
        
    def test_badInputImage2(self):
        data = FMDLData()
        
        data.input_image_np = np.zeros(100)
        
        data.bucketBoundary = (1,2,3,4)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,'maxDiffBetweenAbsBucketEdgeSlopes': 3},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        validResults = getPixel2CmConversionFactor(data)
        
        self.assertFalse(validResults)
        self.assertEqual(data.pixel2CM_conversion_factor , -1)
        self.assertEqual(data.bucketWidthPX, -1)
        self.assertEqual(data.bucketWidthPointsXCord, [])

    
    
    def test_missingConfigItems(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.3406982123851776, 0.059263408184051514, 1.0, 0.8910735845565796)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'measuredBucketWidthCM':300,'maxDiffBetweenAbsBucketEdgeSlopes': 3},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        validResults = getPixel2CmConversionFactor(data)
        
        self.assertFalse(validResults)
        self.assertEqual(data.pixel2CM_conversion_factor , -1)
        self.assertEqual(data.bucketWidthPX, -1)
        self.assertEqual(data.bucketWidthPointsXCord, [])
    
    
    
    def test_badMeasuredBucketWidthCM(self):
        data = FMDLData()

        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.3406982123851776, 0.059263408184051514, 1.0, 0.8910735845565796)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':-0, 
                'maxDiffBetweenAbsBucketEdgeSlopes':3},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),

                ])
        
        validResults = getPixel2CmConversionFactor(data)
        
        self.assertFalse(validResults)
        self.assertEqual(data.pixel2CM_conversion_factor , -1)
        self.assertEqual(data.bucketWidthPX, -1)
        self.assertEqual(data.bucketWidthPointsXCord, [])
    
    
    def test_outputs(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.3406982123851776, 0.059263408184051514, 1.0, 0.8910735845565796)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,'maxDiffBetweenAbsBucketEdgeSlopes': 3},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        validResults = getPixel2CmConversionFactor(data)
        
        self.assertTrue(validResults)
        self.assertTrue(np.allclose(data.pixel2CM_conversion_factor, (0.5639097744360902), atol=1e-04))
        self.assertAlmostEqual(data.bucketWidthPX, 532)
        self.assertEqual(data.bucketWidthPointsXCord[0], 38)
        self.assertEqual(data.bucketWidthPointsXCord[1], 570)
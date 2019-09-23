###############################################################################
# Testing the FMDLAlgoUtils.getDetectedBucketWidthAndHeightInPixels method. 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --None
###############################################################################

from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
from utils.FMDLAlgoUtils import getDetectedBucketWidthAndHeightInPixels
import unittest
import numpy as np


class Test_getDetectedBucketWidthAndHeightInPixels(unittest.TestCase):

    testFilePath = 'tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png'
    with open(testFilePath, "rb") as imageFile:
            f = imageFile.read()
            byteArrayImage = bytearray(f)
    
    def test_missingInputImage(self):
        data = FMDLData()
        
        data.bucketBoundary = (1,2,3,4)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        results = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertEqual(results , (-1, -1, -1, ()))

    
    
    def test_badInputImage1(self):
        data = FMDLData()
        
        data.load_image('tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png')
        
        data.bucketBoundary = (1,2,3,4)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        results = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertEqual(results , (-1, -1, -1, ()))
        
        
        
    def test_badInputImage2(self):
        data = FMDLData()
        
        data.input_image_np = np.zeros(100)
        
        data.bucketBoundary = (1,2,3,4)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        results = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertEqual(results , (-1, -1, -1, ()))



    def test_missingConfigItems(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.3406982123851776, 0.059263408184051514, 1.0, 0.8910735845565796)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'measuredBucketWidthCM':300,},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        results = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertEqual(results , (-1, -1, -1, ()))

    
    
    def test_missingBucketBoundary(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
    
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        results = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertEqual(results , (-1, -1, -1, ()))

    
    
    def test_badBucketBoundary(self):
        data = FMDLData()
        
    
        data.config = Config(
                    {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                    [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
       
        
        data.load_image(self.byteArrayImage)
        
        
        data.bucketBoundary = (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0)
        results1 = getDetectedBucketWidthAndHeightInPixels(data)
        
        
        data.bucketBoundary = (0, -0.8910735845565796, 0, 1.0)
        results2 = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertEqual(results1 , (-1, -1, -1, ()))
        self.assertEqual(results2 , (-1, -1, -1, ()))



    def test_badMeasuredBucketWidthCM(self):
        data = FMDLData()

        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.3406982123851776, 0.059263408184051514, 1.0, 0.8910735845565796)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':-0,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        results = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertEqual(results , (-1, -1, -1, ()))
    
    
    
    def test_bucketBoundaryWithWrongAspectRatio(self):
        data = FMDLData()

        
        data.load_image(self.byteArrayImage)
        
        
        data.bucketBoundary = (4,2,1,3)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        results = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertEqual(results , (-1, -1, -1, ()))


    
    
    def test_badCaseBoundary(self):
        data = FMDLData()
        
    
        data.config = Config(
                    {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                    [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
       
        
        data.load_image(self.byteArrayImage)
        
        
        data.bucketBoundary = (0.3406982123851776, 0.059263408184051514, 1.0, 0.8910735845565796)

        data.bucketBoundary = (4,2,1,3)

        results = getDetectedBucketWidthAndHeightInPixels(data)

        self.assertEqual(results , (-1, -1, -1, ()))



    def test_badEffectiveWidthModifier(self):
        data = FMDLData()
        
    
        data.config = Config(
                    {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":-0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                    [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', -1, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
       
        
        data.load_image(self.byteArrayImage)
        
        
        data.bucketBoundary = (0.3406982123851776, 0.059263408184051514, 1.0, 0.8910735845565796)

        data.bucketBoundary = (4,2,1,3)

        results = getDetectedBucketWidthAndHeightInPixels(data)

        self.assertEqual(results , (-1, -1, -1, ()))



    def test_missingEffectiveWidthModifier(self):
        data = FMDLData()
        
    
        data.config = Config(
                    {'minBoundingBoxAspectRatio':1.5,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":-0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                    [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', -1, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
       
        
        data.load_image(self.byteArrayImage)
        
        
        data.bucketBoundary = (0.3406982123851776, 0.059263408184051514, 1.0, 0.8910735845565796)

        data.bucketBoundary = (4,2,1,3)

        data.config.effectiveWidthYcoordMultiplier = None

        results = getDetectedBucketWidthAndHeightInPixels(data)

        self.assertEqual(results , (-1, -1, -1, ()))
    
    
    
    def test_outputs_noCase(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.36322835087776184, 0.13493812084197998, 1.0, 0.8854146003723145)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        detected_bucketWidth_inPixels_withCase, detected_bucketWith_inPixels_fromBucketAlone,\
     detected_bucketHeight_inPixels, bucketWidthPointsXCord = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertAlmostEqual(detected_bucketWidth_inPixels_withCase, 481)
        self.assertAlmostEqual(detected_bucketWith_inPixels_fromBucketAlone, 481)
        self.assertAlmostEqual(detected_bucketHeight_inPixels, 306)
        self.assertAlmostEqual(bucketWidthPointsXCord[0], 86)
        self.assertAlmostEqual(bucketWidthPointsXCord[1], 567)



    def test_outputs_withCase1(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.20768868923187256, 0.15239688754081726, 0.869536280632019, 0.8369340896606445)

        data.caseBoundary = (0.19773945212364197, 0.27012908458709717, 0.37515637278556824, 0.7303502559661865)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        detected_bucketWidth_inPixels_withCase, detected_bucketWith_inPixels_fromBucketAlone,\
     detected_bucketHeight_inPixels, bucketWidthPointsXCord = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertAlmostEqual(detected_bucketWidth_inPixels_withCase, 365)
        self.assertAlmostEqual(detected_bucketWith_inPixels_fromBucketAlone, 438)
        self.assertAlmostEqual(detected_bucketHeight_inPixels, 317)
        self.assertAlmostEqual(bucketWidthPointsXCord[0], 136)
        self.assertAlmostEqual(bucketWidthPointsXCord[1], 501)



    def test_outputs_withCase_caseMissing(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.20768868923187256, 0.15239688754081726, 0.869536280632019, 0.8369340896606445)

        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        detected_bucketWidth_inPixels_withCase, detected_bucketWith_inPixels_fromBucketAlone,\
     detected_bucketHeight_inPixels, bucketWidthPointsXCord = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertAlmostEqual(detected_bucketWidth_inPixels_withCase, 438)
        self.assertAlmostEqual(detected_bucketWith_inPixels_fromBucketAlone, 438)
        self.assertAlmostEqual(detected_bucketHeight_inPixels, 317)
        self.assertAlmostEqual(bucketWidthPointsXCord[0], 98)
        self.assertAlmostEqual(bucketWidthPointsXCord[1], 536)



    def test_outputs_withCase_emptyCase(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.20768868923187256, 0.15239688754081726, 0.869536280632019, 0.8369340896606445)

        data.caseBoundary = []
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        detected_bucketWidth_inPixels_withCase, detected_bucketWith_inPixels_fromBucketAlone,\
     detected_bucketHeight_inPixels, bucketWidthPointsXCord = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertAlmostEqual(detected_bucketWidth_inPixels_withCase, 438)
        self.assertAlmostEqual(detected_bucketWith_inPixels_fromBucketAlone, 438)
        self.assertAlmostEqual(detected_bucketHeight_inPixels, 317)
        self.assertAlmostEqual(bucketWidthPointsXCord[0], 98)
        self.assertAlmostEqual(bucketWidthPointsXCord[1], 536)



    def test_outputs_withCase_badCase1(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.20768868923187256, 0.15239688754081726, 0.869536280632019, 0.8369340896606445)

        data.caseBoundary = (0.19773945212364197, 0.37515637278556824, 0.7303502559661865)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        detected_bucketWidth_inPixels_withCase, detected_bucketWith_inPixels_fromBucketAlone,\
     detected_bucketHeight_inPixels, bucketWidthPointsXCord = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertAlmostEqual(detected_bucketWidth_inPixels_withCase, 438)
        self.assertAlmostEqual(detected_bucketWith_inPixels_fromBucketAlone, 438)
        self.assertAlmostEqual(detected_bucketHeight_inPixels, 317)
        self.assertAlmostEqual(bucketWidthPointsXCord[0], 98)
        self.assertAlmostEqual(bucketWidthPointsXCord[1], 536)



    def test_outputs_badCase2(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.20768868923187256, 0.15239688754081726, 0.869536280632019, 0.8369340896606445)

        data.caseBoundary = (0.19773945212364197, 0.37515637278556824, 0.7303502559661865)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        detected_bucketWidth_inPixels_withCase, detected_bucketWith_inPixels_fromBucketAlone,\
     detected_bucketHeight_inPixels, bucketWidthPointsXCord = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertAlmostEqual(detected_bucketWidth_inPixels_withCase, 438)
        self.assertAlmostEqual(detected_bucketWith_inPixels_fromBucketAlone, 438)
        self.assertAlmostEqual(detected_bucketHeight_inPixels, 317)
        self.assertAlmostEqual(bucketWidthPointsXCord[0], 98)
        self.assertAlmostEqual(bucketWidthPointsXCord[1], 536)



    def test_outputs_withCase_badCase3(self):
        data = FMDLData()
        
        data.load_image(self.byteArrayImage)
        
        data.bucketBoundary = (0.20768868923187256, 0.15239688754081726, 0.869536280632019, 0.8369340896606445)

        data.caseBoundary = (0.7303502559661865, 0.27012908458709717, 0.37515637278556824, 0.19773945212364197)
        
        data.config = Config(
                {'minBoundingBoxAspectRatio':1,'maxBoundingBoxAspectRatio': 3,'measuredBucketWidthCM':300,"effectiveWidthYcoordMultiplier":0.5,"maxDiffBetweenAbsBucketEdgeSlopes":5},
                [
                        ('measuredBucketWidthCM', 10, 100000, 'float'),
                        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
                        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
                        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
                ])
        
        detected_bucketWidth_inPixels_withCase, detected_bucketWith_inPixels_fromBucketAlone,\
     detected_bucketHeight_inPixels, bucketWidthPointsXCord = getDetectedBucketWidthAndHeightInPixels(data)
        
        self.assertAlmostEqual(detected_bucketWidth_inPixels_withCase, 438)
        self.assertAlmostEqual(detected_bucketWith_inPixels_fromBucketAlone, 438)
        self.assertAlmostEqual(detected_bucketHeight_inPixels, 317)
        self.assertAlmostEqual(bucketWidthPointsXCord[0], 98)
        self.assertAlmostEqual(bucketWidthPointsXCord[1], 536)
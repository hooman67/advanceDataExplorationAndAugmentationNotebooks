###############################################################################
# Testing the utils.Config class 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --None
###############################################################################

from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
import unittest

class Test_Configs(unittest.TestCase):
    
    def test_loadConfigs_noMissingItems(self):
        shovelConfig = {
            "measuredBucketWidthCM":              300,
            "boxDetectorScoreThresholdBucket":    0.5,
            "boxDetectorScoreThresholdMatInside": 0.5,
            "boxDetectorScoreThresholdCase":      0.5,
            "roiDelineatorScoreThreshold":        0.1,
            "minContourArea":                     13000,
            "closingKernelSize":                  7,
            "closingIterations":                  1,
            "erosionKernelSize":                  7,
            "erosionIterations":                  1,
            "roiBoundaryPointsReductionFactor":   0.01,
            "minBoundingBoxAspectRatio":          1.5,
            "maxBoundingBoxAspectRatio":          3,
            "minObjectsRequired":                 [["bucket"]],
            "minBoundingBoxAspectRatio":          1,
            "maxBoundingBoxAspectRatio":          5,
            "intersectingRoiMaxIterations":       5,
            "intersectingRoiStepSize":            0.001,
            "effectiveWidthYcoordMultiplier":     0.5,
            "maxDiffBetweenAbsBucketEdgeSlopes":  5
         }
        
        
        config = Config(shovelConfig, FMDLData.expected_configs)
        
        self.assertTrue(config.is_valid())
        self.assertEqual(config.missing_configs, [])
        
        
        
    def test_loadConfigs_noMissingItems_stringMinObjectsRequired(self):
            shovelConfig = {
                "measuredBucketWidthCM":              300,
                "boxDetectorScoreThresholdBucket":    0.5,
                "boxDetectorScoreThresholdMatInside": 0.5,
                "boxDetectorScoreThresholdCase":      0.5,
                "roiDelineatorScoreThreshold":        0.1,
                "minContourArea":                     13000,
                "closingKernelSize":                  7,
                "closingIterations":                  1,
                "erosionKernelSize":                  7,
                "erosionIterations":                  1,
                "roiBoundaryPointsReductionFactor":   0.01,
                "minBoundingBoxAspectRatio":          1.5,
                "maxBoundingBoxAspectRatio":          3,
                'minObjectsRequired':                 "[['bucket']]",
                "minBoundingBoxAspectRatio":          1,
                "maxBoundingBoxAspectRatio":          5,
                "intersectingRoiMaxIterations":       5,
                "intersectingRoiStepSize":            0.001,
                "effectiveWidthYcoordMultiplier":     0.5,
                "maxDiffBetweenAbsBucketEdgeSlopes":  5
             }
            
            
            config = Config(shovelConfig, FMDLData.expected_configs)
            
            self.assertTrue(config.is_valid())
            self.assertEqual(config.missing_configs, [])


        
    def test_loadConfigs_missingItems(self):
        shovelConfig = {
            'measuredBucketWidthCM':              300,
            'boxDetectorScoreThresholdBucket':    0.5,
            'boxDetectorScoreThresholdMatInside': 0.5,
            'minBoundingBoxAspectRatio':          1.5,
            'maxBoundingBoxAspectRatio':          3,
            'minObjectsRequired':                 [[]],
            }
        
        configsToBeMissed = [
                'roiDelineatorScoreThreshold',
                'minContourArea',
                'closingKernelSize',
                'closingIterations',
                'erosionKernelSize',
                'erosionIterations',
                'roiBoundaryPointsReductionFactor']
        
        
        config = Config(shovelConfig, FMDLData.expected_configs)
        
        self.assertFalse(config.is_valid())
        
        for conf in configsToBeMissed:
            self.assertIn(conf, config.missing_configs)
            
            
    def test_loadConfigs_wrongType(self):
        shovelConfig = {
            "measuredBucketWidthCM":              '300',
            "boxDetectorScoreThresholdBucket":    0.5,
            "boxDetectorScoreThresholdMatInside": 0.5,
            "boxDetectorScoreThresholdCase":      0.5,
            "roiDelineatorScoreThreshold":        0.1,
            "minContourArea":                     13000,
            "closingKernelSize":                  7,
            "closingIterations":                  1,
            "erosionKernelSize":                  7,
            "erosionIterations":                  1,
            "roiBoundaryPointsReductionFactor":   0.01,
            "minBoundingBoxAspectRatio":          1.5,
            "maxBoundingBoxAspectRatio":          3,
            "minObjectsRequired":                 [["bucket"]],
            "minBoundingBoxAspectRatio":          1,
            "maxBoundingBoxAspectRatio":          5,
            "intersectingRoiMaxIterations":       5,
            "intersectingRoiStepSize":            0.001,
            "effectiveWidthYcoordMultiplier":     0.5,
            "maxDiffBetweenAbsBucketEdgeSlopes":  5
         }

        config = Config(shovelConfig, FMDLData.expected_configs)

        self.assertTrue(config.is_valid())


    def test_loadConfigs_wrongValue(self):
        shovelConfig = {
            'measuredBucketWidthCM':              'sdfdsf',
            'boxDetectorScoreThresholdBucket':    0.5,
            'boxDetectorScoreThresholdMatInside': 0.5,
            'roiDelineatorScoreThreshold':        0.5,
            'minContourArea':                     8000,
            'closingKernelSize':                  7,
            'closingIterations':                  1,
            'erosionKernelSize':                  7,
            'erosionIterations':                  1,
            'roiBoundaryPointsReductionFactor':   0.01,
            'minBoundingBoxAspectRatio':          1.5,
            'maxBoundingBoxAspectRatio':          3,
            'minObjectsRequired':                 [['bucket']],
            "maxDiffBetweenAbsBucketEdgeSlopes":  5
            }
        
        configsToBeMissed = ['measuredBucketWidthCM']
        
        
        config = Config(shovelConfig, FMDLData.expected_configs)
        
        self.assertFalse(config.is_valid())
        
        for conf in configsToBeMissed:
            self.assertIn(conf, config.missing_configs)
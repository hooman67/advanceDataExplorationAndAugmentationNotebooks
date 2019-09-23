###############################################################################
# Testing the FMDLAlgo.__init__ method. 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --NETWORK_LOAD_TIME_THRESHOLD
###############################################################################

from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
from utils.FMDLAlgoUtils import getBestBoundaryToCropWith
import numpy as np
import unittest


class Test_GetBestBoundaryToCropWith(unittest.TestCase):
    def test_missingBothBoundaries(self):
        data = FMDLData()
        
        data.config = Config(
                {'minObjectsRequired':[['bucket']]},
                [('minObjectsRequired',[ [[]], [['bucket']], [['matInside']],  [['matInside', 'bucket']], [['bucket','matInside']] ])])
        
        
        getBestBoundaryToCropWith(data)
    
        
        self.assertEqual(data.bestBoundary,())
        
    
    def test_missingBucketBoundaryNoMatInsideInConfig(self):
        data = FMDLData()
        
        data.config = Config(
                {'minObjectsRequired':[['bucket']]},
                [('minObjectsRequired',[ [[]], [['bucket']], [['matInside']],  [['matInside', 'bucket']], [['bucket','matInside']] ])])
        
    
        data.matInsideBoundary = (0.36415937542915344, 0.050836026668548584, 0.9990353584289551, 0.9131188988685608)
        
        getBestBoundaryToCropWith(data)
    
        
        self.assertEqual(data.bestBoundary,())
        
        
    def test_missingBucketBoundaryWithMatInsideInConfig(self):
        data = FMDLData()

        data.config = Config(
                {'minObjectsRequired':[['bucket', 'matInside']]},
                [('minObjectsRequired',[ [[]], [['bucket']], [['matInside']],  [['matInside', 'bucket']], [['bucket','matInside']] ])])


        data.matInsideBoundary = (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0)

        getBestBoundaryToCropWith(data)

        self.assertTrue(np.allclose(data.bucketBoundary, (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0), atol=1e-04))
        self.assertTrue(np.allclose(data.bestBoundary, (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0), atol=1e-04))



    def test_missingMatInsideBoundary(self):
        data = FMDLData()
        
        data.config = Config(
                {'minObjectsRequired':[['bucket']]},
                [('minObjectsRequired',[ [[]], [['bucket']], [['matInside']],  [['matInside', 'bucket']], [['bucket','matInside']] ])])
        
    
        data.bucketBoundary = (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0)
        
        getBestBoundaryToCropWith(data)
        
        self.assertTrue(np.allclose(data.bestBoundary, (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0), atol=1e-04))
        
        
    def test_missingMinObjectsRequired(self):
        data = FMDLData()
        
        data.bucketBoundary = (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0)
        data.matInsideBoundary = (0.36415937542915344, 0.050836026668548584, 0.9990353584289551, 0.9131188988685608)
        
        getBestBoundaryToCropWith(data)
    
        
        self.assertEqual(data.bestBoundary,())
        
        
    def test_wrongMinObjectsRequired(self):
        data = FMDLData()
        
        data.config = Config(
                {'minObjectsRequired':[['matInside']]},
                [('minObjectsRequired',[ [[]], [['bucket']], [['matInside']],  [['matInside', 'bucket']], [['bucket','matInside']] ])])
        
        data.bucketBoundary = (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0)
        
        getBestBoundaryToCropWith(data)
    
        
        self.assertEqual(data.bestBoundary,())
        
        
    def test_outputs(self):
        data = FMDLData()
        
        data.config = Config({"minObjectsRequired": [["bucket"]]},[('minObjectsRequired',[ [[]], [['bucket']], [['matInside']],  [['matInside', 'bucket']], [['bucket','matInside']] ])])
        
        data.bucketBoundary = (0.3406982123851776, 0.059263408184051514, 1.0, 0.8910735845565796)
        data.matInsideBoundary = (0.36415937542915344, 0.050836026668548584, 0.9990353584289551, 0.9131188988685608)
        
        getBestBoundaryToCropWith(data)

        print(data.bestBoundary)

        self.assertTrue(np.allclose(data.bestBoundary, (0.36415937542915344, 0.059263408184051514, 0.9990353584289551, 0.8910735845565796), atol=1e-04))
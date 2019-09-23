from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
import cv2
from utils.FMDLAlgoUtils import getRoiContour
import unittest
import numpy as np


class Test_FMDLAlgoUtils_getRoiContour(unittest.TestCase):

            
    postProcessedRoi = cv2.imread('tests/networksAndImages-forUnitTesting/closedAndEroded1_roi'\
        'ActualSize_FMDL_2018.04.30_19.11.20.png')
    postProcessedRoi = cv2.cvtColor(postProcessedRoi, cv2.COLOR_BGR2GRAY)
    
    
    def test_outputs(self):
        data = FMDLData()
        
        data.postProcessed_roi_actualSize = self.postProcessedRoi
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01},
                    [('minContourArea', 10, 100000, 'integer'),('roiBoundaryPointsReductionFactor',\
                     0.001, 0.1, 'float'),])
        

        validResults = getRoiContour(data)
        
        self.assertTrue(validResults)


        testroiBoundaryContour = np.load('tests/networksAndImages-forUnitTesting/roiBoundaryContour_FMDL_2018.04.30_19.11.20.npy') 

        print("\nthis is the predicted roi_boundary_contour point\n")
        print(data.roi_boundary_contour)
        print("\nthis is the groundTruth testroiBoundaryContour point\n")
        print(testroiBoundaryContour)
     
        self.assertTrue(np.allclose(data.roi_boundary_contour, testroiBoundaryContour, atol=1))
        
        
        
    def test_missingPostProcessedRoi(self):
        data = FMDLData()
        
        data.config = Config(
                    {'minContourArea': 8000,'roiBoundaryPointsReductionFactor': 0.01},
                    [('minContourArea', 10, 100000, 'integer'),('roiBoundaryPointsReductionFactor',\
                     0.001, 0.1, 'float'),])
        
        validResults = getRoiContour(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.approximated_roi_boundary), 0)
        
        
        
    def test_badPostProcessedRoi(self):
        data = FMDLData()
        
        data.postProcessed_roi_actualSize = np.zeros((100,1), int)
    
        data.config = Config(
                    {
                            'minContourArea':                     8000,
                            'roiBoundaryPointsReductionFactor':   0.01
                    },
                    [('minContourArea', 10, 100000, 'integer'),('roiBoundaryPointsReductionFactor',\
                     0.001, 0.1, 'float'),])
        
        validResults = getRoiContour(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.approximated_roi_boundary), 0)
        
 
        
    def test_missingConfigItems(self):
        data = FMDLData()
        
        data.postProcessed_roi_actualSize = self.postProcessedRoi
        
        validResults = getRoiContour(data)
        
        self.assertFalse(validResults)
        self.assertEqual(len(data.approximated_roi_boundary), 0)
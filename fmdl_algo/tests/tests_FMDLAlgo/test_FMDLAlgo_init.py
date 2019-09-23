###############################################################################
# Testing the FMDLAlgo.__init__ method. 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --NETWORK_LOAD_TIME_THRESHOLD
###############################################################################

from datetime import datetime, timedelta

#Values to Test Against
NETWORK_LOAD_TIME_THRESHOLD = timedelta(seconds=50)

import pytest
from fmdlAlgo.FMDLAlgo import FMDLAlgo
import unittest

class Test_FMDLAlgoInit(unittest.TestCase):

    def test_networkLoad_Exception_badNetwork_boxDetector(self):
        with pytest.raises(Exception):
            FMDLAlgo(
                'tests/networksAndImages-forUnitTesting/BadBoxDetectorNetwork.pb',
                'tests/networksAndImages-forUnitTesting/roiDelineator.h5',
                debugMode=True)
    
            
    def test_networkLoad_Exception_badNetwork_roiDelineator(self):
        with pytest.raises(Exception):
            FMDLAlgo(
                'tests/networksAndImages-forUnitTesting/bbDetector.pb',
                'tests/networksAndImages-forUnitTesting/BadRoiDelineatorNework.h5',
                debugMode=True)
            
    
    def test_networkLoad_Exception(self):
        with pytest.raises(Exception):
            FMDLAlgo(
                'tests/networksAndImages-forUnitTesting/BadBoxDetectorNetwork.pb',
                'tests/networksAndImages-forUnitTesting/BadRoiDelineatorNework.h5',
                debugMode=True)
    
            
    def test_networkLoad_Exception_networkNotFound_boxDetector(self):
        with pytest.raises(Exception):
            FMDLAlgo(
                '',
                'tests/networksAndImages-forUnitTesting/roiDelineator.h5',
                debugMode=True)
            
    
    def test_networkLoad_Exception_networkNotFound_roiDelineator(self):
        with pytest.raises(Exception):
            FMDLAlgo(
                'tests/networksAndImages-forUnitTesting/bbDetector.pb',
                '',
                debugMode=True)
    
    
    def test_networkLoad_Time_withDebugMode(self):
        global fmdlAlgo_Hydraulic
    
        algoLoadStartTime = datetime.now()
    
        fmdlAlgo_Hydraulic = FMDLAlgo(
            'tests/networksAndImages-forUnitTesting/bbDetector.pb',
            'tests/networksAndImages-forUnitTesting/roiDelineator.h5',
            debugMode=True)
        
        algoLoadTime = datetime.now() - algoLoadStartTime
        
        self.assertIsNotNone(fmdlAlgo_Hydraulic)
        self.assertLess(algoLoadTime, NETWORK_LOAD_TIME_THRESHOLD)


        
    def test_networkLoad_Time_withoutDebugMode(self):
        global fmdlAlgo_Hydraulic
    
        algoLoadStartTime = datetime.now()
    
        fmdlAlgo_Hydraulic = FMDLAlgo(
            'tests/networksAndImages-forUnitTesting/bbDetector.pb',
            'tests/networksAndImages-forUnitTesting/roiDelineator.h5')
        
        algoLoadTime = datetime.now() - algoLoadStartTime
        
        self.assertIsNotNone(fmdlAlgo_Hydraulic)
        self.assertLess(algoLoadTime, NETWORK_LOAD_TIME_THRESHOLD)
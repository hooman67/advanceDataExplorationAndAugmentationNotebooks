###############################################################################
# Testing the FMDLAlgo.__init__ method. 
# Tests, corrupted networks, missing networks, and network load time.
# Required values to test against:
#   --NETWORK_LOAD_TIME_THRESHOLD
###############################################################################

from fmdlAlgo.FMDLAlgo import FMDLData
import unittest
import numpy as np


class Test_FMDLData_loadImage(unittest.TestCase):
       
    
    def test_png1chan(self):
        data = FMDLData()
        
        testFilePath = 'tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.png'
        with open(testFilePath, "rb") as imageFile:
                f = imageFile.read()
                byteArrayImage = bytearray(f)
                
        
        data.load_image(byteArrayImage)
        
        imageToTestAgainst = np.load('tests/networksAndImages-forUnitTesting/png_FMDL_2018.04.30_19.11.20.npy')
        testImHeight,testImWidth, _ = imageToTestAgainst.shape

        self.assertTrue(data.is_valid_image)
        self.assertEqual(data.imageWidthPx, testImWidth)
        self.assertEqual(data.imageHeightPx, testImHeight)

        self.assertTrue(np.allclose(data.input_image_np, imageToTestAgainst, atol=1e-04))
        
        
        
    def test_png3chan(self):
        data = FMDLData()
        
        testFilePath = 'tests/networksAndImages-forUnitTesting/3Chan_FMDL_2018.04.30_19.11.20.png'
        with open(testFilePath, "rb") as imageFile:
                f = imageFile.read()
                byteArrayImage = bytearray(f)
                
        
        data.load_image(byteArrayImage)
        
        imageToTestAgainst = np.load('tests/networksAndImages-forUnitTesting/png_FMDL_2018.04.30_19.11.20.npy')
        testImHeight,testImWidth, _ = imageToTestAgainst.shape

        self.assertTrue(data.is_valid_image)
        self.assertEqual(data.imageWidthPx, testImWidth)
        self.assertEqual(data.imageHeightPx, testImHeight)
        self.assertTrue(np.allclose(data.input_image_np, imageToTestAgainst, atol=1e-04))
        
        
        
    def test_jpg1Chan(self):
        data = FMDLData()
        
        testFilePath = 'tests/networksAndImages-forUnitTesting/FMDL_2018.04.30_19.11.20.jpg'
        with open(testFilePath, "rb") as imageFile:
                f = imageFile.read()
                byteArrayImage = bytearray(f)
                
        
        data.load_image(byteArrayImage)

        imageToTestAgainst = np.load('tests/networksAndImages-forUnitTesting/jpg_FMDL_2018.04.30_19.11.20.npy')
        testImHeight,testImWidth, _ = imageToTestAgainst.shape

        self.assertTrue(data.is_valid_image)
        self.assertEqual(data.imageWidthPx, testImWidth)
        self.assertEqual(data.imageHeightPx, testImHeight)

        self.assertTrue(np.allclose(data.input_image_np, imageToTestAgainst, atol=1e-04))
        
        
        
    def test_missingInputImage(self):
        data = FMDLData()
        
        byteArrayImage = []
                
        
        data.load_image(byteArrayImage)
        
        self.assertFalse(data.is_valid_image)
        
        
        
    def test_badInputImage(self):
        data = FMDLData()
        
        testFilePath = 'tests/networksAndImages-forUnitTesting/bad_FMDL_2018.04.30_19.11.20.png'
        with open(testFilePath, "rb") as imageFile:
                f = imageFile.read()
                byteArrayImage = bytearray(f)
                
        
        data.load_image(byteArrayImage)
                
        
        data.load_image(byteArrayImage)
        
        self.assertFalse(data.is_valid_image)


    def test_rbg(self):
        data = FMDLData()
        
        testFilePath = 'tests/networksAndImages-forUnitTesting/RGB_FMDL_2018.04.30_19.11.20.png'
        with open(testFilePath, "rb") as imageFile:
                f = imageFile.read()
                byteArrayImage = bytearray(f)
                
        
        data.load_image(byteArrayImage)
        
        imageToTestAgainst = np.load('tests/networksAndImages-forUnitTesting/png_FMDL_2018.04.30_19.11.20.npy')
        testImHeight,testImWidth, _ = imageToTestAgainst.shape

        self.assertTrue(data.is_valid_image)
        self.assertEqual(data.imageWidthPx, testImWidth)
        self.assertEqual(data.imageHeightPx, testImHeight)

        self.assertTrue(np.allclose(data.input_image_np, imageToTestAgainst, atol=1e-04))
        
        
        
    def test_bgr(self):
        data = FMDLData()
        
        testFilePath = 'tests/networksAndImages-forUnitTesting/BGR_FMDL_2018.04.30_19.11.20.png'
        with open(testFilePath, "rb") as imageFile:
                f = imageFile.read()
                byteArrayImage = bytearray(f)
                
        
        data.load_image(byteArrayImage)
        
        imageToTestAgainst = np.load('tests/networksAndImages-forUnitTesting/png_FMDL_2018.04.30_19.11.20.npy')
        testImHeight,testImWidth, _ = imageToTestAgainst.shape

        self.assertTrue(data.is_valid_image)
        self.assertEqual(data.imageWidthPx, testImWidth)
        self.assertEqual(data.imageHeightPx, testImHeight)

        self.assertTrue(np.allclose(data.input_image_np, imageToTestAgainst, atol=1e-04))
        
        
        
    def test_gray(self):
        data = FMDLData()
        
        testFilePath = 'tests/networksAndImages-forUnitTesting/GRAY_FMDL_2018.04.30_19.11.20.png'
        with open(testFilePath, "rb") as imageFile:
                f = imageFile.read()
                byteArrayImage = bytearray(f)
                
        
        data.load_image(byteArrayImage)
        
        imageToTestAgainst = np.load('tests/networksAndImages-forUnitTesting/png_FMDL_2018.04.30_19.11.20.npy')
        testImHeight,testImWidth, _ = imageToTestAgainst.shape

        self.assertTrue(data.is_valid_image)
        self.assertEqual(data.imageWidthPx, testImWidth)
        self.assertEqual(data.imageHeightPx, testImHeight)
        self.assertTrue(np.allclose(data.input_image_np, imageToTestAgainst, atol=1e-04))
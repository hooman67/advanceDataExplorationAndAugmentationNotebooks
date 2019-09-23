from fmdlAlgo.FMDLAlgo import FMDLData
from utils.config import Config
from utils.FMDLAlgoUtils import getWidthFromBucketAndCase
import unittest
import numpy as np


class Test_getWidthFromBucketAndCase(unittest.TestCase):

    goundTruth_bucketWidth_inPixels_withCase = 365
    goundTruth_bucketLeftEdge = ((98, 417), (173, 180))
    goundTruth_bucketRightEdge = ((536, 417), (467, 180))
    goundTruth_bucketMidEdge = ((136, 298), (501, 298))

    goundTruth_maxDiffBetweenAbsBucketEdgeSlopes = 3

    goundTruth_bucketBoundary = (0.20768868923187256, 0.15239688754081726, 0.869536280632019, 0.8369340896606445)

    goundTruth_caseBoundary = (0.19773945212364197, 0.27012908458709717, 0.37515637278556824, 0.7303502559661865)

    goundTruth_effectiveWidthYcoordMultiplier = 0.5

    goundTruth_im_height = 480
    goundTruth_im_width = 640

    goundTruth_bucketWidthPointsXCord = [136, 501]




    def test_outputs_withCase1(self):
        detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord = \
            getWidthFromBucketAndCase(
                self.goundTruth_bucketBoundary,
                self.goundTruth_caseBoundary,
                self.goundTruth_effectiveWidthYcoordMultiplier,
				self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes,
                self.goundTruth_im_height, self.goundTruth_im_width
            )


        self.assertAlmostEqual(
            detected_bucketWidth_inPixels_withCase,
            self.goundTruth_bucketWidth_inPixels_withCase
        )


        self.assertAlmostEqual(
            
            bucketLeftEdge,
            self.goundTruth_bucketLeftEdge
        )

        self.assertAlmostEqual(
            bucketRightEdge,
            self.goundTruth_bucketRightEdge
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            self.goundTruth_bucketMidEdge
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            self.goundTruth_bucketMidEdge
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord[0],
            self.goundTruth_bucketWidthPointsXCord[0]
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord[1],
            self.goundTruth_bucketWidthPointsXCord[1]
        )




    def test_missingBucketBoundary(self):
        detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord = \
            getWidthFromBucketAndCase(
                [],
                self.goundTruth_caseBoundary,
                self.goundTruth_effectiveWidthYcoordMultiplier,
				self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes,
                self.goundTruth_im_height, self.goundTruth_im_width
            )
        
        self.assertAlmostEqual(
            detected_bucketWidth_inPixels_withCase,
            None
        )


        self.assertAlmostEqual(
            bucketLeftEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketRightEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord,
            ()
        )



    def test_badBucketBoundary1(self):
        detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord = \
            getWidthFromBucketAndCase(
                "badd",
                self.goundTruth_caseBoundary,
                self.goundTruth_effectiveWidthYcoordMultiplier,
				self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes,
                self.goundTruth_im_height, self.goundTruth_im_width
            )



        print("\n\n\n******************************")
        print('detected_bucketWidth_inPixels_withCase')
        print(detected_bucketWidth_inPixels_withCase)
        print("******************************\n\n\n")
        
        self.assertAlmostEqual(
            detected_bucketWidth_inPixels_withCase,
            None
        )


        self.assertAlmostEqual(
            bucketLeftEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketRightEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord,
            ()
        )



    def test_badBucketBoundary2(self):
        detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord = \
            getWidthFromBucketAndCase(
                (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0),
                self.goundTruth_caseBoundary,
                self.goundTruth_effectiveWidthYcoordMultiplier,
				self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes,
                self.goundTruth_im_height, self.goundTruth_im_width
            )
        
        self.assertAlmostEqual(
            detected_bucketWidth_inPixels_withCase,
            None
        )


        self.assertAlmostEqual(
            bucketLeftEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketRightEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord,
            ()
        )



    def test_badBucketBoundary3(self):
        detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord = \
            getWidthFromBucketAndCase(
                (0, -0.8910735845565796, 0, 1.0),
                self.goundTruth_caseBoundary,
                self.goundTruth_effectiveWidthYcoordMultiplier,
				self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes,
                self.goundTruth_im_height, self.goundTruth_im_width
            )
        
        self.assertAlmostEqual(
            detected_bucketWidth_inPixels_withCase,
            None
        )


        self.assertAlmostEqual(
            bucketLeftEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketRightEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord,
            ()
        )


    def test_missingCaseBoundary(self):
        detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord = \
            getWidthFromBucketAndCase(
                self.goundTruth_bucketBoundary,
                [],
                self.goundTruth_effectiveWidthYcoordMultiplier,
				self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes,
                self.goundTruth_im_height, self.goundTruth_im_width
            )
        
        self.assertAlmostEqual(
            detected_bucketWidth_inPixels_withCase,
            None
        )


        self.assertAlmostEqual(
            bucketLeftEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketRightEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord,
            ()
        )


    def test_badCaseBoundary1(self):
        detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord = \
            getWidthFromBucketAndCase(
                self.goundTruth_bucketBoundary,
                "badd",
                self.goundTruth_effectiveWidthYcoordMultiplier,
				self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes,
                self.goundTruth_im_height, self.goundTruth_im_width
            )
        
        self.assertAlmostEqual(
            detected_bucketWidth_inPixels_withCase,
            None
        )


        self.assertAlmostEqual(
            bucketLeftEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketRightEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord,
            ()
        )



    def test_badCaseBoundary2(self):
        detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord = \
            getWidthFromBucketAndCase(
                self.goundTruth_bucketBoundary,
                (0.3406982123851776, 0.8910735845565796, 0.059263408184051514, 1.0),
                self.goundTruth_effectiveWidthYcoordMultiplier,
				self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes,
                self.goundTruth_im_height, self.goundTruth_im_width
            )
        
        self.assertAlmostEqual(
            detected_bucketWidth_inPixels_withCase,
            None
        )


        self.assertAlmostEqual(
            bucketLeftEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketRightEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord,
            ()
        )



    def test_badCaseBoundary3(self):
        detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord = \
            getWidthFromBucketAndCase(
                self.goundTruth_bucketBoundary,
                (0, -0.8910735845565796, 0, 1.0),
                self.goundTruth_effectiveWidthYcoordMultiplier,
				self.goundTruth_maxDiffBetweenAbsBucketEdgeSlopes,
                self.goundTruth_im_height, self.goundTruth_im_width
            )
        
        self.assertAlmostEqual(
            detected_bucketWidth_inPixels_withCase,
            None
        )


        self.assertAlmostEqual(
            bucketLeftEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketRightEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketMidEdge,
            ()
        )

        self.assertAlmostEqual(
            bucketWidthPointsXCord,
            ()
        )
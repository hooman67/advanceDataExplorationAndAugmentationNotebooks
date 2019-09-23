import io, cv2, logging

import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq

from PIL import Image as PilImage

import utils.visualizationUtils as visUtils


#All bounding boxes are define with 4 numbers
EXPECTED_BOUNDING_BOX_LENGTH = 4

#All edges (lines) are define with 2 numbers.
EXPECTED_EDGE_LENGTH = 2


class NetworkLoadException(Exception):
    def __init__(self, networkName):
        message = "Could not load the specified network:  " + networkName
        super(NetworkLoadException, self).__init__(message)


#Takes a bytearray image, converts it to Pillow image, converts it to npArray
def loadBytearrayImageIntoNp(image_bytearray):
    logger = logging.getLogger(__name__)
    validImage = False
    output_np_image = []
    im_width = -1
    im_height = -1

    try:
        image = PilImage.open(io.BytesIO(image_bytearray)).convert("RGB")

        (im_width, im_height) = image.size

        output_np_image = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        validImage = True;
    except:
        validImage = False;
        logger.error("Could not convert the provided input image from bytearray to numpy array.\
         A possible cause is that the input image is corrupted.\n")

    return validImage, output_np_image, im_width, im_height



def getMinBoundary(bucketBoundary, matInsideBoundary):
    out = [
            max(bucketBoundary[0], matInsideBoundary[0]),
            max(bucketBoundary[1], matInsideBoundary[1]),
            min(bucketBoundary[2], matInsideBoundary[2]),
            min(bucketBoundary[3], matInsideBoundary[3])
           ]

    return tuple(out)


def validateCroppedImage(data):
    try:
        im_height, im_width, _ = data.input_image_np.shape
        crop_im_height, crop_im_width, _ = data.input_image_np_cropped.shape
        
        if \
            crop_im_height > 0 and\
            crop_im_width > 0  and\
            crop_im_height <= im_height and\
            crop_im_width <= im_width:
                return True
        else:
            return False
        
    except (AttributeError, ValueError):
        return False
    
    
def validateUncroppedImage(data):
    
    try:
        crop_roi_height, crop_roi_width = data.roi_roidDelineatorSize.shape
        uncrop_roi_height, uncrop_roi_width = data.roi_actualSize.shape
        
        if \
            uncrop_roi_height > 0 and\
            uncrop_roi_width > 0  and\
            crop_roi_height > 0 and\
            crop_roi_width > 0  and\
            crop_roi_height <= uncrop_roi_height and\
            crop_roi_width <= uncrop_roi_width:
                return True
        else:
            return False
        
    except (AttributeError, ValueError):
        return False


def cropImage(data):
    logger = logging.getLogger(__name__)

    global EXPECTED_BOUNDING_BOX_LENGTH
    
    boundary = data.bestBoundary
    if len(boundary) != EXPECTED_BOUNDING_BOX_LENGTH:
        logger.warning("The cropImage method did not receive a valid bestBoundary, not\
         cropping image\n")
        return False
    
    imageIn = data.input_image_np
    if  len(imageIn) == 0:
        logger.error("input image provided to FMDLAlgoUtils cropImage method is empty. \
            Not cropping image. We will early return!!!\n")
        return False
    
    
    try:
        ymin, xmin, ymax, xmax = boundary
    
        im_height, im_width, _ = imageIn.shape
    
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    
        (left, right, top, bottom) = (int(round(left)), int(round(right)), int(round(top)), int(round(bottom)))
    
        outIm = imageIn[top:bottom, left:right]
    
    except (AttributeError, ValueError):
        logger.error("Found a suitable best boundary to crop with but still failed to crop.\
         The boundary used was%s\n\n",
                              boundary)
        return False


    logger.debug("The cropImage method received a boundary with left = %s , right = %s , top = %s ,\
     bottom = %s , \nThe input image has height = %s , width = %s\n\n",\
      left, right, top, bottom, im_height, im_width)


    data.input_image_np_cropped = outIm
    
    return validateCroppedImage(data)



def uncropImage(data):
    logger = logging.getLogger(__name__)
    global EXPECTED_BOUNDING_BOX_LENGTH
    
    boundary = data.bestBoundary
    if len(boundary) != EXPECTED_BOUNDING_BOX_LENGTH:
        logger.warning("The uncropImage method did not receive a valid bestBoundary,\
         not uncropping image\n")
        return False
    
    imageIn = data.input_image_np
    if  len(imageIn) == 0:
        logger.error("input image provided to FMDLAlgoUtils uncropImage method is empty.\
         Not uncropImage image. We will early return!!!\n")
        return False
    
    
    unetPred = data.roi_roidDelineatorSize
    if  len(unetPred) == 0:
        logger.error("roi_roidDelineatorSize (U-Net prediction binary image) provided to \
            FMDLAlgoUtils uncropImage method is empty. Not uncropImage image. We will early\
             return!!!\n")
        return False
    
    
    try:
        ymin, xmin, ymax, xmax = boundary
    
        im_height, im_width, _ = imageIn.shape
    
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    
        (left, right, top, bottom) = (int(round(left)), int(round(right)), int(round(top)),\
         int(round(bottom)))
    
        unetActualSize = np.zeros((imageIn.shape[0], imageIn.shape[1]), bool)
    
        unetCroppedSize = cv2.resize(unetPred, ((right - left), (bottom - top)))
    
        unetActualSize[top:bottom, left:right] = unetCroppedSize
    
        unetActualSize.dtype='uint8'
        
    except (cv2.error, TypeError):
        logger.error("Found a boundary, image, and roi to uncrop with but still failed to uncrop.\
         The boundary used was%s\n\n",
                              boundary)
        return False
        
    
    data.roi_actualSize = unetActualSize

    return validateUncroppedImage(data)



def getBestBoundaryToCropWith(data):
    logger = logging.getLogger(__name__)
    global EXPECTED_BOUNDING_BOX_LENGTH
    bestBoundary = ()
    
    bucketBoundary = data.bucketBoundary
    matInsideBoundary = data.matInsideBoundary

    try:    
        minObjectsRequired = data.config.minObjectsRequired
    except AttributeError:
        logger.error("minObjectsRequired is missing from the parameters supplied to FMDLAlgoUtils\
         getBestBoundaryToCropWith method. We will early return because bestBoundary to crop with\
          cannot be calculated\n")
        return
    
    if bucketBoundary and matInsideBoundary and len(bucketBoundary) == 4 and \
    len(matInsideBoundary) == EXPECTED_BOUNDING_BOX_LENGTH:
        bestBoundary = getMinBoundary(bucketBoundary, matInsideBoundary)
        logger.debug("The getBestBoundaryToCropWith method proposed to croped the image with\
         overlap of bucketBoundary and matInsideBoundary, calculated to be:\n%s\n\n", bestBoundary)

    else:
        logger.debug("The getBestBoundaryToCropWith method did not receive both bucketBoundary \
            and matInsideBoundary.\n For bucketBoundary it received:\n%s\n\nFor matInside Boundary\
             it received:\n%s\n\nWe will try to use the bucketBoundary alone if available and\
              specified in minObjectsRequired config item.\n",
                     bucketBoundary, matInsideBoundary)
        
        for item in minObjectsRequired:
            if 'bucket' in item and bucketBoundary:
                bestBoundary = bucketBoundary
                logger.debug("The getBestBoundaryToCropWith method proposed to croped the image\
                 with bucketBoundary, previousy found to be:\n%s\n\n", bestBoundary)

            if 'matInside' in item and matInsideBoundary:
                bestBoundary = matInsideBoundary
                logger.debug("The getBestBoundaryToCropWith method proposed to croped the image\
                 with matInsideBoundary, previousy found to be:\n%s\n\n", matInsideBoundary)

                if not bucketBoundary:
                    data.bucketBoundary = bestBoundary
                    logger.info("Could not find a bucketBoundary, but config specifies that we can\
                     use matInside Boundary instead. From this point on matInside boundary is\
                      treated as bucketBoundary!, it was found to be:\n%s\n\n", data.bucketBoundary)
    data.bestBoundary = bestBoundary



def getPixel2CmConversionFactor(data):
    logger = logging.getLogger(__name__)
    pixel2CM_conversion_factor = -1
    detected_bucketWidth_inPixels = -1
    boxDetectorPredictionValid = False
    bucketWidthPointsXCord = []


    try:
        measuredBucketWidthCM = data.config.measuredBucketWidthCM
        minBoundingBoxAspectRatio = data.config.minBoundingBoxAspectRatio
        maxBoundingBoxAspectRatio = data.config.maxBoundingBoxAspectRatio
    
    except AttributeError:
        logger.error("measuredBucketWidthCM or minBoundingBoxAspectRatio or "\
            "maxBoundingBoxAspectRatio is missing from the parameters supplied to FMDLAlgoUtils "\
            "getPixel2CmConversionFactor method. We will early return!!!\n")

        pixel2CM_conversion_factor = -1
        detected_bucketWidth_inPixels_withCase = -1
        bucketWidthPointsXCord = []
        data.bucketWidthPointsXCord = bucketWidthPointsXCord
        return False


    detected_bucketWidth_inPixels_withCase, detected_bucketWith_inPixels_fromBucketAlone,\
     detected_bucketHeight_inPixels, bucketWidthPointsXCord = getDetectedBucketWidthAndHeightInPixels(data)

    if detected_bucketWidth_inPixels_withCase <= 0 or detected_bucketWith_inPixels_fromBucketAlone\
     <= 0 or detected_bucketHeight_inPixels <= 0 or not len(bucketWidthPointsXCord) == 2:
        logger.warning("The getPixel2CmConversionFactor method calculated the width in pixel or "\
            "height in pixel to be <= 0, or couldnt find the bucketWidthPointsXCord; not goint to"\
         " calculate pixel2CM_conversion_factor\n")

        pixel2CM_conversion_factor = -1
        detected_bucketWidth_inPixels_withCase = -1
        bucketWidthPointsXCord = []
        data.bucketWidthPointsXCord = bucketWidthPointsXCord
        return False
    

    detected_bucket_aspectRatio = float(detected_bucketWith_inPixels_fromBucketAlone)\
     / float(detected_bucketHeight_inPixels)


    pixel2CM_conversion_factor = measuredBucketWidthCM / detected_bucketWidth_inPixels_withCase

    logger.debug("The getPixel2CmConversionFactor method calculated:\ndetected_bucketWidth_inPixels"\
        "= %s\ndetected_bucketHeight_inPixels = %s\ndetected_bucket_aspectRatio = "\
        "%s\npixel2CM_conversion_factor = %s\n\n",
                 detected_bucketWidth_inPixels_withCase, detected_bucketHeight_inPixels,
                 detected_bucket_aspectRatio, pixel2CM_conversion_factor)

    boxDetectorPredictionValid = minBoundingBoxAspectRatio < detected_bucket_aspectRatio <\
     maxBoundingBoxAspectRatio

    logger.debug("Detected Bounding Box passed aspect ratio check: %s\n", boxDetectorPredictionValid)


    if(pixel2CM_conversion_factor <= 0 or detected_bucketWidth_inPixels_withCase <= 0):
        pixel2CM_conversion_factor = -1
        detected_bucketWidth_inPixels_withCase = -1
        bucketWidthPointsXCord = []
        data.bucketWidthPointsXCord = bucketWidthPointsXCord
        boxDetectorPredictionValid = False

        
    data.pixel2CM_conversion_factor = pixel2CM_conversion_factor
    data.bucketWidthPX = detected_bucketWidth_inPixels_withCase
    data.bucketWidthPointsXCord = bucketWidthPointsXCord

    return boxDetectorPredictionValid



def validateAllDataFor_getDetectedBucketWidthAndHeightInPixels(data):
    logger = logging.getLogger(__name__)
    global EXPECTED_BOUNDING_BOX_LENGTH

    allDataIsValid = False
    imageIn = []
    im_height = None
    im_width = None
    effectiveWidthYcoordMultiplier = None
    bucketBoundary = ()
    maxDiffBetweenAbsBucketEdgeSlopes = None

    try:
        imageIn = data.input_image_np
    except AttributeError:
        logger.error("input_image_np is missing from the parameters "\
            "supplied to FMDLAlgoUtils validateAllDataFor_getDetectedBucketWidthAndHeightInPixels method.\n")
        return allDataIsValid, imageIn, im_height,\
     im_width, effectiveWidthYcoordMultiplier, bucketBoundary, maxDiffBetweenAbsBucketEdgeSlopes

    if imageIn == []:
        logger.error("input image provided to FMDLAlgoUtils validateAllDataFor_getDetectedBucketWidthAndHeightInPixels"\
            " method is empty. We will early return!!!\n")
        return allDataIsValid, imageIn, im_height,\
     im_width, effectiveWidthYcoordMultiplier, bucketBoundary, maxDiffBetweenAbsBucketEdgeSlopes


    try:
        im_height, im_width, _ = imageIn.shape
    except (AttributeError, ValueError):
        logger.error("there was something wrong with the image input"\
            " supplied to FMDLAlgoUtils validateAllDataFor_getDetectedBucketWidthAndHeightInPixels method. We will early return!!!\n")
        return allDataIsValid, imageIn, im_height,\
     im_width, effectiveWidthYcoordMultiplier, bucketBoundary, maxDiffBetweenAbsBucketEdgeSlopes


    try:
        effectiveWidthYcoordMultiplier = data.config.effectiveWidthYcoordMultiplier
    except AttributeError:
        logger.error("effectiveWidthYcoordMultiplier is missing from"\
            " the parameters supplied to FMDLAlgoUtils validateAllDataFor_getDetectedBucketWidthAndHeightInPixels method.\n")
        return allDataIsValid, imageIn, im_height,\
     im_width, effectiveWidthYcoordMultiplier, bucketBoundary, maxDiffBetweenAbsBucketEdgeSlopes



    bucketBoundary = data.bucketBoundary
    if len(bucketBoundary) != EXPECTED_BOUNDING_BOX_LENGTH:
        logger.warning("The getPixel2CmConversionFactor method did not"\
            " receive a valid bucketBoundary, not calculating pixel2CM_conversion_factor\n")
        return allDataIsValid, imageIn, im_height,\
     im_width, effectiveWidthYcoordMultiplier, bucketBoundary, maxDiffBetweenAbsBucketEdgeSlopes


    try:
        maxDiffBetweenAbsBucketEdgeSlopes = data.config.maxDiffBetweenAbsBucketEdgeSlopes
    except AttributeError:
        logger.error("maxDiffBetweenAbsBucketEdgeSlopes is missing"\
            " from the parameters supplied to FMDLAlgoUtils validateAllDataFor_getDetectedBucketWidthAndHeightInPixels method.\n")
        return allDataIsValid, imageIn, im_height,\
     im_width, effectiveWidthYcoordMultiplier, bucketBoundary, maxDiffBetweenAbsBucketEdgeSlopes


    allDataIsValid = True

    return allDataIsValid, imageIn, im_height,\
     im_width, effectiveWidthYcoordMultiplier, bucketBoundary, maxDiffBetweenAbsBucketEdgeSlopes



def getDetectedBucketWidthAndHeightInPixels(data):
    logger = logging.getLogger(__name__)
    global EXPECTED_BOUNDING_BOX_LENGTH
    global EXPECTED_EDGE_LENGTH

    detected_bucketWidth_inPixels_withCase = ()
    bucketLeftEdge = ()
    bucketRightEdge = ()
    bucketMidEdge = ()
    bucketWidthPointsXCord_withCase = ()


    allDataIsValid, imageIn, im_height,\
     im_width, effectiveWidthYcoordMultiplier, bucketBoundary, maxDiffBetweenAbsBucketEdgeSlopes =\
      validateAllDataFor_getDetectedBucketWidthAndHeightInPixels(data)

    if not allDataIsValid:
        logger.error("getDetectedBucketWidthAndHeightInPixels method"\
            " did not receive all the data it needed. We will early return!!!\n")
        return -1, -1, -1, ()



    detected_bucketWith_inPixels_fromBucketAlone, detected_bucketHeight_inPixels, bucketWidthPointsXCord_fromBucketAlone =\
     getWidthAndHeightFromBucketAlone(bucketBoundary, im_height, im_width)


    caseBoundary = data.caseBoundary
    if len(caseBoundary) == EXPECTED_BOUNDING_BOX_LENGTH:
        try:
            detected_bucketWidth_inPixels_withCase, bucketLeftEdge, bucketRightEdge, bucketMidEdge, bucketWidthPointsXCord_withCase = \
            getWidthFromBucketAndCase(
                bucketBoundary,
                caseBoundary,
                effectiveWidthYcoordMultiplier,
                maxDiffBetweenAbsBucketEdgeSlopes,
                im_height,
                im_width
            )

        except TypeError:
            logger.error("The getDetectedBucketWidthAndHeightInPixels received a caseBoundary,"\
                " BUT it was NOT valid. we will use just the bucketObject in pixel2CM_conversion_factor\n")

            detected_bucketWidth_inPixels_withCase = detected_bucketWith_inPixels_fromBucketAlone
            bucketWidthPointsXCord_withCase = bucketWidthPointsXCord_fromBucketAlone
        
        
        logger.info("The getDetectedBucketWidthAndHeightInPixels received a valid caseBoundary,"\
            " we will use the case object in calculating pixel2CM_conversion_factor\n")

    else:
        detected_bucketWidth_inPixels_withCase = detected_bucketWith_inPixels_fromBucketAlone
        bucketWidthPointsXCord_withCase = bucketWidthPointsXCord_fromBucketAlone
        
        logger.info("The getDetectedBucketWidthAndHeightInPixels did not recieve a valid"\
            " caseBoundary, we will use just the bucketObject pixel2CM_conversion_factor\n")


    # if we failed to use case to calculate the width, just return the one from bucket
    if not detected_bucketWidth_inPixels_withCase or \
    detected_bucketWidth_inPixels_withCase <= 0:

        detected_bucketWidth_inPixels_withCase = detected_bucketWith_inPixels_fromBucketAlone
        bucketWidthPointsXCord_withCase = bucketWidthPointsXCord_fromBucketAlone
        
        logger.info("The getDetectedBucketWidthAndHeightInPixels failed to calulated the width "\
            "using caseBoundary, we ended up using just the bucketObject pixel2CM_conversion_factor\n")

    else:
        if len(bucketLeftEdge) == EXPECTED_EDGE_LENGTH and\
         len(bucketRightEdge) == EXPECTED_EDGE_LENGTH and\
         len(bucketMidEdge) == EXPECTED_EDGE_LENGTH:
            data.bucketLeftEdge = bucketLeftEdge
            data.bucketRightEdge = bucketRightEdge
            data.bucketMidEdge = bucketMidEdge

            logger.debug("The getDetectedBucketWidthAndHeightInPixels validated the bucket edges"\
                " and added them to the data object\n")

        else:
            logger.debug("The getDetectedBucketWidthAndHeightInPixels failed to validate the bucket edges"\
                " and did NOT add them to the data object\n")


    logger.debug("The getDetectedBucketWidthAndHeightInPixels successfully returned the following:"\
        "\ndetected_bucketWidth_inPixels_withCase : %s\ndetected_bucketWith_inPixels_fromBucketAlone:"\
        " %s\ndetected_bucketHeight_inPixels: %s\nbucketWidthPointsXCord_withCase: %s\n",
        str(detected_bucketWidth_inPixels_withCase),
        str(detected_bucketWith_inPixels_fromBucketAlone),
        str(detected_bucketHeight_inPixels),
        str(bucketWidthPointsXCord_withCase))

   
    return detected_bucketWidth_inPixels_withCase, detected_bucketWith_inPixels_fromBucketAlone,\
     detected_bucketHeight_inPixels, bucketWidthPointsXCord_withCase



def getWidthFromBucketAndCase(bucketBoundary, caseBoundary, effectiveWidthYcoordMultiplier, maxDiffBetweenAbsBucketEdgeSlopes, im_height, im_width):
    logger = logging.getLogger(__name__)


    left_buck, right_buck, top_buck, bottom_buck, validBuck = \
    convertBoundaryFromRatio2Pixel(bucketBoundary, im_height, im_width)
    left_case, right_case, top_case, bottom_case, validCase = \
    convertBoundaryFromRatio2Pixel(caseBoundary, im_height, im_width)

    if not validBuck or not validCase:
        logger.warning("The getWidthFromBucketAndCase method faild to"\
            " find appropriate points on bucket edges using the case"\
            " object to calculate bucket edge equations. We are gonna"\
            " fall back to using just the bucketObject instead.\n")
        return None, (), (), (), ()



    resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope,\
     rightEdge_bias = getBucketEdgeLineEquations(
        (left_buck,  bottom_buck),
        (left_case,  bottom_case),
        (right_buck, bottom_buck),
        (right_case, bottom_case),
        maxDiffBetweenAbsBucketEdgeSlopes
    )

    if not resultAreValid:
      logger.error("The getWidthFromBucketAndCase method failed to"\
        " getLineSlopeAndBias of bucket edgtes. We are gonna"\
            " fall back to using just the bucketObject instead.\n")
      return None, (), (), (), ()

    else:
        logger.debug("The getWidthFromBucketAndCase method validated the bucket edge equations.\n")



    ycord2MeasureBucketWidhtAt = int(round( bottom_case + (bottom_buck - bottom_case) *\
     effectiveWidthYcoordMultiplier))

    if not ycord2MeasureBucketWidhtAt or ycord2MeasureBucketWidhtAt <= 0\
     or ycord2MeasureBucketWidhtAt > im_width:

      logger.error("The getWidthFromBucketAndCase method caclculated"\
        " a ycord2MeasureBucketWidhtAt that did not pass our validation."\
        " We are gonna fall back to using just the bucketObject instead.\n")

      return None, (), (), (), ()


    try:
        xcordLeft =  int(round(solve4XCord(leftEdge_slope, leftEdge_bias, ycord2MeasureBucketWidhtAt)))
        xcordRight = int(round(solve4XCord(rightEdge_slope, rightEdge_bias, ycord2MeasureBucketWidhtAt)))
    except:
        logger.error("The getWidthFromBucketAndCase method caclculated"\
        " the bucket edge line equations but failed to solve for the "\
        "x coordinates of the specified y location to measure the bucketWidth"\
        " at. We are gonna fall back to using just the bucketObject instead.\n")
        return None, (), (), (), ()

    if not (xcordRight > 0 and xcordLeft > 0 and xcordRight > xcordLeft):
        logger.error("The getWidthFromBucketAndCase method caclculated"\
        " the bucket edge line equations but failed to solve for the "\
        "x coordinates of the specified y location to measure the bucketWidth"\
        " at. We are gonna fall back to using just the bucketObject instead.\n")
        return None, (), (), (), ()


    logger.debug("getWidthFromBucketAndCase returned valid results.\nxcordRight: %s\nxcordLeft:"\
        " %s\n", str(xcordRight),str(xcordLeft))


    return (xcordRight - xcordLeft), ((left_buck, bottom_buck), (left_case, bottom_case)),\
     ((right_buck, bottom_buck), (right_case, bottom_case)), ( (xcordLeft, ycord2MeasureBucketWidhtAt),\
      (xcordRight, ycord2MeasureBucketWidhtAt) ), (xcordLeft, xcordRight)



def getBucketEdgeLineEquations(leftEdgePoint1, leftEdgePoint2,\
 rightEdgePoint1, rightEdgePoint2, maxDiffBetweenAbsBucketEdgeSlopes):
    logger = logging.getLogger(__name__)

    resultAreValid = False
    leftEdge_slope = None
    leftEdge_bias = None
    rightEdge_slope = None
    rightEdge_bias = None


    try:
        leftEdge_slope, leftEdge_bias = getLineSlopeAndBias(
            leftEdgePoint1,
            leftEdgePoint2
        )

        rightEdge_slope, rightEdge_bias = getLineSlopeAndBias(
            rightEdgePoint1,
            rightEdgePoint2
        )

    except:
        logger.error("The getBucketEdgeLineEquations method failed to"\
            " getLineSlopeAndBias of bucket edgtes, not"\
            " calculating pixel2CM_conversion_factor. Algo will early return\n")
        return resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope, rightEdge_bias



    if leftEdge_slope is None or leftEdge_bias is None or\
     rightEdge_slope is None or rightEdge_bias is None or\
     (leftEdge_slope == 0 and leftEdge_bias == 0) or\
     (rightEdge_slope == 0 and rightEdge_bias == 0) :

        logger.error("The getBucketEdgeLineEquations method failed to"\
            " getLineSlopeAndBias of bucket edgtes. The returned values were empty"\
            " or all zeros. Not calculating pixel2CM_conversion_factor."\
            " Algo will early return\n")
        return resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope, rightEdge_bias


    try:
        if abs( abs(leftEdge_slope) - abs(rightEdge_slope) )\
         > maxDiffBetweenAbsBucketEdgeSlopes:
            logger.error("The getBucketEdgeLineEquations method calculated"\
                " slopes for left and right bucket edges that were too"\
                " different. Not calculating pixel2CM_conversion_factor."\
                " Algo will early return\n")
            return resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope, rightEdge_bias

    except TypeError:
        logger.error("The getBucketEdgeLineEquations method calculated"\
                " slopes for left and right bucket edges but typeError"\
                " was thrown when validating them. Not calculating pixel2CM_conversion_factor."\
                " Algo will early return\n")

        return resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope, rightEdge_bias


    logger.info("The getBucketEdgeLineEquations method successfully"\
        " validated to the slopes of the calculated bucket edge equations.\n")

    resultAreValid = True

    return resultAreValid, leftEdge_slope, leftEdge_bias, rightEdge_slope, rightEdge_bias



def convertBoundaryFromRatio2Pixel(boundary, im_height, im_width):
    logger = logging.getLogger(__name__)

    try:
        ymin, xmin, ymax, xmax = boundary

        (left, right, top, bottom) = (
            int(round(xmin * im_width)),
            int(round(xmax * im_width)),
            int(round(ymin * im_height)),
            int(round(ymax * im_height)))

    except:
        logger.error("The convertBoundaryFromRatio2Pixel method threw an"\
            " exception and failed to convert coordinates from ratio to pixel.\n")

        return -1,-1,-1,-1, False


    if left >= right or top >= bottom:
        logger.error("The convertBoundaryFromRatio2Pixel method failed"\
            " to convert coordinates from ratio to pixel.\n")
        return -1,-1,-1,-1, False

    else:
        return left, right, top, bottom, True



def getWidthAndHeightFromBucketAlone(bucketBoundary, im_height, im_width):
    logger = logging.getLogger(__name__)
    global EXPECTED_BOUNDING_BOX_LENGTH

    if len(bucketBoundary) != EXPECTED_BOUNDING_BOX_LENGTH:
        logger.error("The getWidthAndHeightFromBucketAlone method did not receive a valid\
         bucketBoundary, not calculating pixel2CM_conversion_factor\n")
        return -1, -1, ()


    left, right, top, bottom, validConversion = convertBoundaryFromRatio2Pixel(bucketBoundary,\
     im_height, im_width)

    if not validConversion:
        logger.error("The getWidthAndHeightFromBucketAlone method observed that the calculated\
         bucket boundary is not suitable for calculating bucketWidth in pixels.\n")
        return -1, -1, ()
    

    detected_bucketWidth_inPixels = (right - left)
    detected_bucketHeight_inPixels = (bottom - top)
    bucketWidthPointsXCord = [left, right]

    return detected_bucketWidth_inPixels, detected_bucketHeight_inPixels, bucketWidthPointsXCord



def applyClosing(binary_image, closingKernelSize, closingIterations):
    logger = logging.getLogger(__name__)
    
    if len(binary_image) == 0:
        logger.warning("input binary_image provided to FMDLAlgoUtils applyClosing method is empty.\
         Won't apply Closing. This may not actually cause us to early return!!!\n")
        return binary_image
    
    if closingIterations <= 0:
        logger.info("closingIterations provided to FMDLAlgoUtils applyClosing method means no\
         Closing will be applied. Supplied value was: %s\n\n",
                    closingIterations)
        return binary_image


    try:
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (closingKernelSize, closingKernelSize))
    
        binary_image_closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, element,\
         iterations=closingIterations)
        
    except (cv2.error, TypeError):
        logger.error("Received a binary_image, and valid closingIterations but opencv failed to\
         apply closing.Supplied closingKernelSize = %s,  closingIterations = %s\n\n",
                              closingKernelSize, closingIterations)
        return binary_image
    
    
    logger.debug("Closing successfully applied.\n")
    return binary_image_closed




def applyErosion(binary_image, erosionKernelSize, erosionIterations):
    logger = logging.getLogger(__name__)
    
    if len(binary_image) == 0:
        logger.warning("input binary_image provided to FMDLAlgoUtils applyErosion method is empty.\
         Won't apply erossion. This WILL cause us to early return!!!\n")
        return binary_image
    
    if erosionKernelSize <= 0:
        logger.info("erosionKernelSize provided to FMDLAlgoUtils applyErosion method means no\
         erosion will be applied. Supplied value was: %s\n\n", erosionKernelSize)
        return binary_image

        
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (erosionKernelSize, erosionKernelSize))

        binary_image_eroded = cv2.erode(binary_image, kernel, iterations=erosionIterations)
    
    except (cv2.error, TypeError):
        logger.error("Received a binary_image, and valid erosionIterations but opencv failed to\
         apply erosion.Supplied erisionKernelSize = %s,  erosionIterations = %s\n\n",
                              erosionKernelSize, erosionIterations)
        return binary_image
        
    logger.debug("Erosion successfully applied.\n")
    return binary_image_eroded



def get_contours(image):
    logger = logging.getLogger(__name__)
    
    if len(image) == 0:
        logger.error("input image provided to FMDLAlgoUtils get_contours method is empty. Won't\
         find any contours. We will early return!!!\n")
        return []
        
    try:
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,\
         offset=(0, 0))
        
    except (cv2.error, TypeError):
        logger.error("Received an image but opencv failed to find any contours. We will early \
            return!!!\n")
        return []
    
    logger.debug("get_contours returned contours.\n")
    return contours


def postProcessRoi(data):
    logger = logging.getLogger(__name__)
    
    roi_actualSize = data.roi_actualSize
    
    if  len(roi_actualSize) == 0:
        logger.error("roi_actualSize (U-Net prediction binary image) provided to FMDLAlgoUtils \
            getPostProcessedRoiContours via getRoiBoundaryPoints method is empty. We will early\
             return!!!\n")
        return False
    
    
    try:
        closingKernelSize = data.config.closingKernelSize
        closingIterations = data.config.closingIterations
        erosionKernelSize = data.config.erosionKernelSize
        erosionIterations = data.config.erosionIterations
    except AttributeError:
        logger.error(" At least one of the following config items is missing from the parameters\
         supplied to FMDLAlgoUtils getRoiBoundaryPoints method. We will early return!!!\n\
         (closingKernelSize , closingIterations , erosionKernelSize , erosionIterations\n\n")
        return False



    roi_actualSize_closed = applyClosing(roi_actualSize, closingKernelSize, closingIterations)
    
    if data.debugMode:
        roi_actualSize_closed.dtype='uint8'
        data.debug_info.append({'description': 'resized roi after Closing',\
            'image': visUtils.encodeImageAsBase64(cv2.cvtColor(roi_actualSize_closed*255,\
             cv2.COLOR_GRAY2BGR))})

    
    roi_actualSize_closed_eroded = applyErosion(roi_actualSize_closed, erosionKernelSize,\
     erosionIterations)
    
    if data.debugMode:
        roi_actualSize_closed_eroded.dtype='uint8'
        data.debug_info.append({'description': 'resized and closed roi after erosion',\
            'image': visUtils.encodeImageAsBase64(cv2.cvtColor(roi_actualSize_closed_eroded*255,\
             cv2.COLOR_GRAY2BGR))})
        

    data.postProcessed_roi_actualSize = roi_actualSize_closed_eroded
 
    return True



def validateAllDataFor_getRoiBoundaryPoints(data):
    logger = logging.getLogger(__name__)

    allDataIsValid = False
    maxContour = np.empty([], dtype=int)
    roiBoundaryPointsReductionFactor = None
    intersectingRoiMaxIterations = None
    intersectingRoiStepSize = None

    try:
        maxContour = data.roi_boundary_contour
    except AttributeError:
            logger.error("validateAllDataFor_getRoiBoundaryPoints"\
                " method did NOT receive a valid roi_boundary_contour"\
                " inside the data object. This will cause us to reject"\
                " this log.")
            return allDataIsValid, maxContour, roiBoundaryPointsReductionFactor, intersectingRoiMaxIterations, intersectingRoiStepSize
 
    if  maxContour.size == 0:
        logger.error("data.roi_boundary_contour (i.e maxContour) provided to FMDLAlgoUtils "\
            "getRoiBoundaryPoints method is empty. We will early return!!!\n")
        return allDataIsValid, maxContour, roiBoundaryPointsReductionFactor, intersectingRoiMaxIterations, intersectingRoiStepSize

    
    try:
        roiBoundaryPointsReductionFactor = data.config.roiBoundaryPointsReductionFactor
        intersectingRoiMaxIterations = data.config.intersectingRoiMaxIterations
        intersectingRoiStepSize = data.config.intersectingRoiStepSize
        
        logger.debug("getRoiBoundaryPoints received the following config items:\n"\
            "roiBoundaryPointsReductionFactor = %s\nintersectingRoiMaxIterations = %s\n"\
            "intersectingRoiStepSize = %s\n",str(roiBoundaryPointsReductionFactor),\
             str(intersectingRoiMaxIterations), str(intersectingRoiStepSize))

    except AttributeError:
        logger.error(" At least one of the following config items is missing from the parameters "\
            "supplied to FMDLAlgoUtils getRoiBoundaryPoints method. We will early return!!!\n"\
            "(roiBoundaryPointsReductionFactor, intersectingRoiMaxIterations,"\
            "intersectingRoiStepSize\n\n")
        return allDataIsValid, maxContour, roiBoundaryPointsReductionFactor, intersectingRoiMaxIterations, intersectingRoiStepSize


    allDataIsValid = True

    return allDataIsValid, maxContour, roiBoundaryPointsReductionFactor, intersectingRoiMaxIterations, intersectingRoiStepSize



def getRoiBoundaryPoints(data):
    logger = logging.getLogger(__name__)
    

    allDataIsValid, maxContour, roiBoundaryPointsReductionFactor,\
     intersectingRoiMaxIterations, intersectingRoiStepSize = validateAllDataFor_getRoiBoundaryPoints(data)

    if not allDataIsValid:
        logger.error("getRoiBoundaryPoints method did NOT receive the data"\
        " it was expecting. Either roi_boundary_contour or at least one"\
        " of the following config items is missing from the parameters "\
        "supplied to FMDLAlgoUtils getRoiBoundaryPoints method. We"\
        " will early return!!!\n"\
        "(roiBoundaryPointsReductionFactor, intersectingRoiMaxIterations,"\
        "intersectingRoiStepSize\n\n")
        return False

    
    if intersectingRoiMaxIterations > 0:
        for correctionItr in range(intersectingRoiMaxIterations):
            current_roiBoundaryPointsReductionFactor = roiBoundaryPointsReductionFactor - \
            (correctionItr * intersectingRoiStepSize)
         
            approximated_roi_boundary = getApproximatedRoiBoundaryPoints(maxContour,\
             current_roiBoundaryPointsReductionFactor)
            if len(approximated_roi_boundary) <= 0:
                logger.error("getRoiBoundaryPoints method received all configuration items it "\
                    "expected, but getApproximatedRoiBoundaryPoints returned empty."\
                    "Algo will early return!!!\n\n")
                return False
            
            if not areRoiBoundaryPointsIntersecting(approximated_roi_boundary[:,0,:]):
                logger.debug("The findRoiBoundaryPoints method approximated the roi boundary with"\
                    " %s  points. In iteration %s using current_roiBoundaryPointsReductionFactor "\
                    "of %s\n", len(approximated_roi_boundary), str(correctionItr),\
                    str(current_roiBoundaryPointsReductionFactor))
                data.approximated_roi_boundary = approximated_roi_boundary
                return True
            else:
                logger.debug("In findRoiBoundaryPoints method, areRoiBoundaryPointsIntersecting return true. Iteration %s using length of approximated_roi_boundary is "\
                    " %s\n", str(correctionItr), len(approximated_roi_boundary))
            
        logger.warning("The findRoiBoundaryPoints method could not find a reduced set of"\
            "ROI boundary points that do not self-intersect after all iterations.")
        return False

    else:
        logger.debug("The findRoiBoundaryPoints method received an intersectingRoiMaxIterations"\
            "that is not >0. The check to make sure roi is not self intersecting is disabled now.\n")
        approximated_roi_boundary = getApproximatedRoiBoundaryPoints(maxContour,\
         roiBoundaryPointsReductionFactor)
        if len(approximated_roi_boundary) <= 0:
            logger.error("getRoiBoundaryPoints method received all configuration items it expected,"\
                "but getApproximatedRoiBoundaryPoints returned empty. Algo will early return!!!\n\n")
            return False
        else:
            logger.debug("The findRoiBoundaryPoints method approximated the roi boundary with  %s\
              points.\n", len(approximated_roi_boundary))
            data.approximated_roi_boundary = approximated_roi_boundary
            return True



def getApproximatedRoiBoundaryPoints(maxContour, roiBoundaryPointsReductionFactor):
    logger = logging.getLogger(__name__)
    approximated_roi_boundary = []
    
    try:
        epsilon = roiBoundaryPointsReductionFactor * cv2.arcLength(maxContour,True)
        approximated_roi_boundary = cv2.approxPolyDP(maxContour, epsilon, True)
        return approximated_roi_boundary

    except (cv2.error, TypeError):
        logger.error('getApproximatedRoiBoundaryPoints, cv2.approxPolyDP failed. Algo will'\
            'early return!!!\n\n')
        return approximated_roi_boundary



############### For points and lines ##########################
def pointOnLineSegment(A,B,C):
    #checks if Point C is on line segment AB
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def linesIntersect(line1, line2):
    logger = logging.getLogger(__name__)

    doLinesIntersect = True

    A = np.array(line1[0])
    B = np.array(line1[1])

    C = np.array(line2[0])
    D = np.array(line2[1])


    try:
        doLinesIntersect = pointOnLineSegment(A,C,D) != pointOnLineSegment(B,C,D) and pointOnLineSegment(A,B,C) != pointOnLineSegment(A,B,D)

    except:
        logger.debug("The linesIntersect method threw and exception and ended"\
            " up returning true (lines intersect) eventho it couldn't"\
            " actually determine intersection.\n line1 was: %s\n line2 was: %s\n",\
            str(line1), str(line2))

        return True
    

    return doLinesIntersect


def getLineSlopeAndBias(point1, point2):
    points = [point1, point2]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]

    #print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    return m, c


def solve4XCord(slope, bias, yCord):
    return (yCord - bias) / slope
#################################################################



def areRoiBoundaryPointsIntersecting(boundary_points):
    logger = logging.getLogger(__name__)

    logger.debug("The areRoiBoundaryPointsIntersecting received this reduced roi boundary:\n%s\n",\
     str(boundary_points))
    
    if boundary_points is None or len(boundary_points) <= 0 or boundary_points.shape[0] < 3:
        logger.error("The areRoiBoundaryPointsIntersecting received an invalid roi boundary.\n")
        return True


    lines = []

    for i in range(len(boundary_points)-1):
        newLine = (boundary_points[i], boundary_points[i+1])

        for oldLine in lines[:-1]:
            if linesIntersect(oldLine, newLine):
                logger.debug("The areRoiBoundaryPointsIntersecting observed that the reduced ROI\
                 boundary self intersects.\n")
                return True

        lines.append( newLine )
        
    logger.debug("The areRoiBoundaryPointsIntersecting validated that the reduced ROI boundary"\
        " does not self intersect.\n")
    return False



def getRoiContour(data):
    logger = logging.getLogger(__name__)
    
    roi_actualSize_closed_eroded = data.postProcessed_roi_actualSize
    
    if  len(roi_actualSize_closed_eroded) == 0:
        logger.error("roi_actualSize_closed_eroded provided to FMDLAlgoUtils getRoiContour method"\
            " is empty. We will early return!!!\n")
        return False

    
    try:
        minContourArea = data.config.minContourArea
    except AttributeError:
        logger.error(" minContourArea is missing from the parameters supplied to FMDLAlgoUtils "\
            "getRoiContour method. We will early return!!!")
        return False



    contours = get_contours(roi_actualSize_closed_eroded)

    if len(contours) == 0:
        logger.warning("The getRoiContour method received a none empty postProcessed_roi_actualSize"\
            " but, no contours were found. Algo will early return!!!\n")
        return False


    maxContour = max(contours, key=cv2.contourArea)  # get the largest contour area
    logger.debug("The getRoiContour method observed that after closing and erosion, the length of"\
        " the LARGEST contour is: %s\n\n", len(maxContour))
    
    if cv2.contourArea(maxContour) < minContourArea:
        logger.warning("The getRoiContour method observed that after closing and erosion, contour"\
            " area is less than the minimum required\nCalculated contour area was: %s\n while the"\
            " minimum required area was specified as: %s\n\n",
            cv2.contourArea(maxContour), minContourArea)
        return False
 

    logger.debug("The getRoiContour method found a max contour of area %s for the ROI.\n",\
     cv2.contourArea(maxContour))

    data.roi_boundary_contour = maxContour

    return True



def validateApproximatedRoiBoundary(data):
    logger = logging.getLogger(__name__)

    approximated_roi_boundary = data.approximated_roi_boundary 
    
    if len(approximated_roi_boundary) == 0:
        logger.debug("validateApproximatedRoiBoundary observed that the length of "\
            "approximated_roi_boundary is 0, which means the validation of final results was"\
            " UNSUCCESSFUL.\n")
        return False
    
    try:
        dim1, dim2, dim3 = approximated_roi_boundary.shape
        
        if dim1 > 2:
            #just getting rid of the unused dimention
            data.approximated_roi_boundary_2D = approximated_roi_boundary[:,0,:]     
            return True
            
    except (AttributeError, ValueError):
        logger.debug("validateApproximatedRoiBoundary could not verify the shape of"\
            " approximated_roi_boundary, which means the validation of final results was UNSUCCESSFUL.\n")
        return False
    
    
    
def validateBucketBox(data):
    logger = logging.getLogger(__name__)
    global EXPECTED_BOUNDING_BOX_LENGTH
    
    if len(data.bucketBoundary) != EXPECTED_BOUNDING_BOX_LENGTH:
        logger.debug("The validateBucketBox method did not receive valid bucketBox."\
            " We received:\n  %s.\n", data.bucketBoundary) 
        
        data.bucketValid = False
        return False
    else:
        logger.debug("The validateBucketBox method received the following bucketBoundary:\n%s.\n",
            data.bucketBoundary) 
    
        imageIn = data.input_image_np
            
        ymin, xmin, ymax, xmax = data.bucketBoundary 
    
        im_height, im_width, _ = imageIn.shape
    
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    
        (left, right, top, bottom) = (int(round(left)), int(round(right)), int(round(top)), \
            int(round(bottom)))
    
        data.bucketBox = np.array([[left, top],[right, bottom]])
        data.bucketValid = True
        
        logger.debug("The validateBucketBox method returned the following bucketBox:\n  %s.\n",\
         data.bucketBox) 
        
        return True

    
    
def validateMatInsideBox(data):
    logger = logging.getLogger(__name__)
    global EXPECTED_BOUNDING_BOX_LENGTH
    
    if len(data.matInsideBoundary) != EXPECTED_BOUNDING_BOX_LENGTH:
        logger.debug("The validateMatInsideBox method did not receive valid bucketBox."\
            " We received:\n%s.\n", data.matInsideBoundary) 
        
        data.matInsideValid = False
        return False
    else:
        logger.debug("The validateMatInsideBox method received the following bucketBoundary:\n  %s.\n"\
            , data.matInsideBoundary) 
    
        imageIn = data.input_image_np
            
        ymin, xmin, ymax, xmax = data.matInsideBoundary 
    
        im_height, im_width, _ = imageIn.shape
    
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    
        (left, right, top, bottom) = (int(round(left)), int(round(right)), int(round(top)),\
         int(round(bottom)))
    
        data.matInsideBox = np.array([[left, top],[right, bottom]])
        data.matInsideValid = True
        
        logger.debug("The validateMatInsideBox method returned the following bucketBox:\n  %s.\n",\
         data.matInsideBox) 
        
        return True



def validateBucketLines(data):
    logger = logging.getLogger(__name__)
    global EXPECTED_BOUNDING_BOX_LENGTH
    global EXPECTED_EDGE_LENGTH
    

    if len(data.caseBoundary) != EXPECTED_BOUNDING_BOX_LENGTH:
        logger.info("The validateBucketLines method did not receive valid caseBox."\
            " We received:\n%s.\n", data.caseBoundary) 
        data.effectiveWidthValid = False
        return False


    if len(data.bucketLeftEdge) != EXPECTED_EDGE_LENGTH or\
     len(data.bucketRightEdge) != EXPECTED_EDGE_LENGTH or\
     len(data.bucketMidEdge) != EXPECTED_EDGE_LENGTH:
        logger.error("The validateBucketLines method received the following caseBoundary:\n  %s \n"\
            "But the one of buckets edges was empty. We received:\n leftEdge:  %s \n rightEdge:"\
            "  %s \n midEdge:  %s \n", data.caseBoundary, data.bucketLeftEdge, data.bucketRightEdge,\
           data.bucketMidEdge) 
        data.effectiveWidthValid = False
        return False


    data.bucketLeftLine = np.array([[data.bucketLeftEdge[0][0], data.bucketLeftEdge[0][1]],\
        [data.bucketLeftEdge[1][0], data.bucketLeftEdge[1][1]]])
    data.bucketRightLine = np.array([[data.bucketRightEdge[0][0], data.bucketRightEdge[0][1]],\
        [data.bucketRightEdge[1][0], data.bucketRightEdge[1][1]]])
    data.bucketMidLine = np.array([[data.bucketMidEdge[0][0], data.bucketMidEdge[0][1]],\
        [data.bucketMidEdge[1][0], data.bucketMidEdge[1][1]]])


    logger.debug("The validateBucketLines method returned the following:\n leftEdge:"\
        "  %s \n rightEdge:  %s \n midEdge:  %s \n", data.bucketLeftLine, data.bucketRightLine,\
      data.bucketMidLine) 
    data.effectiveWidthValid = True
    return True



def getNumpyAsList(numpyArray, dataIdentifierString):
    logger = logging.getLogger(__name__)
    outputList = []

    logger.debug("getNumpyAsList was called on:  %s  .\n", dataIdentifierString)

    if type(numpyArray).__module__ == np.__name__:
        try:
            outputList = numpyArray.tolist()
        except:
            logger.error("getNumpyAsList recieved an object that seemed like numpy but was not."\
                " This should be investigated.\n")
    else:
        outputList = list(numpyArray)



    if not isinstance(outputList, list):
        logger.error("getNumpyAsList thought that it had successfully converted the object"\
            " it received to python list, but secondary verifications failed, so we returned an"\
            " empty list instead. This should be investigated.\n")

        outputList = []

    return outputList
import logging, cv2, json
import numpy as np
from detectors.BoxDetector import BoxDetector
from detectors.RoiDelineator import RoiDelineator
import utils.FMDLAlgoUtils as utils
import utils.visualizationUtils as visUtils
from utils.config import Config


INITIAL_NEGATIVE_NUMBER = -1

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)



class FMDLData:
    expected_configs = [
        # (config_name, lower bound, upper bound)
        ('measuredBucketWidthCM', 10, 100000, 'float'),
        ('boxDetectorScoreThresholdBucket', 0, 1, 'float'),
        ('boxDetectorScoreThresholdMatInside', 0, 1, 'float'),
        ('boxDetectorScoreThresholdCase', 0, 1, 'float'),
        ('roiDelineatorScoreThreshold', 0, 1, 'float'),
        ('minContourArea', 10, 100000, 'integer'),
        ('closingKernelSize', 1, 20, 'integer'),
        ('closingIterations', 0, 20, 'integer'),
        ('erosionKernelSize', 1, 20, 'integer'),
        ('erosionIterations', 0, 20, 'integer'),
        ('roiBoundaryPointsReductionFactor', 0.001, 0.1, 'float'),
        ('minObjectsRequired',[ [[]], [['bucket']], [['matInside']],  [['matInside', 'bucket']],\
         [['bucket', 'matInside']] ]),
        ('minBoundingBoxAspectRatio', 0.1, 10, 'float'),
        ('maxBoundingBoxAspectRatio', 0.1, 10, 'float'),
        ('intersectingRoiMaxIterations', 0, 100, 'integer'),
        ('intersectingRoiStepSize', 0.00001, 0.1, 'float'),
        ('effectiveWidthYcoordMultiplier', 0, 1, 'float'),
        ('maxDiffBetweenAbsBucketEdgeSlopes', 0, 100, 'float'),
    ]

    def __init__(self, debugMode=False):
        self.debugMode = debugMode
        self.initAllData()



    def initAllData(self):
        self.initAlgorithmInputsData()
        self.initAlgorithmOutputData()
        self.initAlgorithmIntermidiateData()



    def initAlgorithmInputsData(self):
        self.config = []
        self.input_image_np = []
        self.is_valid_image = False
        


    def initAlgorithmOutputData(self):
        global INITIAL_NEGATIVE_NUMBER

        self.imageWidthPx = INITIAL_NEGATIVE_NUMBER
        self.imageHeightPx = INITIAL_NEGATIVE_NUMBER
        self.pixel2CM_conversion_factor = INITIAL_NEGATIVE_NUMBER
        self.bucketWidthPX = INITIAL_NEGATIVE_NUMBER
        self.debug_info = []

        self.valid = False
        self.bucketValid = False
        self.matInsideValid = False
        self.effectiveWidthValid = False

        self.approximated_roi_boundary_2D = np.empty([], dtype=int)
        self.bucketBox = np.empty([], dtype=int)
        self.matInsideBox = np.empty([], dtype=int)
        self.bucketLeftLine = np.empty([], dtype=int)
        self.bucketRightLine = np.empty([], dtype=int)
        self.bucketMidLine = np.empty([], dtype=int)
        self.bucketWidthPointsXCord = np.empty([], dtype=int)



    def clearOutputData(self):
        global INITIAL_NEGATIVE_NUMBER

        self.imageWidthPx = INITIAL_NEGATIVE_NUMBER
        self.imageHeightPx = INITIAL_NEGATIVE_NUMBER
        self.pixel2CM_conversion_factor = INITIAL_NEGATIVE_NUMBER
        self.bucketWidthPX = INITIAL_NEGATIVE_NUMBER
        self.debug_info = []

        self.valid = False
        self.bucketValid = False
        self.matInsideValid = False
        self.effectiveWidthValid = False

        self.approximated_roi_boundary_2D = []
        self.bucketBox = []
        self.matInsideBox = []
        self.bucketLeftLine = []
        self.bucketRightLine = []
        self.bucketMidLine = []
        self.bucketWidthPointsXCord = []


    def initAlgorithmIntermidiateData(self):
        global INITIAL_NEGATIVE_NUMBER

        self.bucketBoundary = ()
        self.bucketScore = INITIAL_NEGATIVE_NUMBER
        self.matInsideBoundary = ()
        self.matInsideScore = INITIAL_NEGATIVE_NUMBER
        self.bestBoundary = ()

        self.caseScore = INITIAL_NEGATIVE_NUMBER
        self.caseBoundary = ()
        self.bucketLeftEdge = ()
        self.bucketRightEdge = ()
        self.bucketMidEdge = ()
        
        self.input_image_np_cropped = []
        self.roi_roidDelineatorSize = []
        self.roi_actualSize = []
        self.postProcessed_roi_actualSize = []
        self.roi_boundary_contour = np.empty([], dtype=int)
        self.approximated_roi_boundary = []
        
        

    def load_image(self, input_image):
        self.is_valid_image, self.input_image_np, self.imageWidthPx,  self.imageHeightPx =\
         utils.loadBytearrayImageIntoNp(input_image)



    def as_dict(self, debug = False):
        data_dict = {
            'valid':                              self.valid,
            'pixel2CM_conversion_factor':         self.pixel2CM_conversion_factor,
            'detected_bucketWidth_inPixels':      self.bucketWidthPX,
            'imageWidthPx':                       self.imageWidthPx,
            'imageHeightPx':                      self.imageHeightPx,
            'bucket_valid':                       self.bucketValid,
            'matInside_valid':                    self.matInsideValid,
            'effective_width_calculations_valid': self.effectiveWidthValid,
            'approximated_roi_boundary':          utils.getNumpyAsList(self.approximated_roi_boundary_2D, 'approximated_roi_boundary_2D'),
            'approximated_bucket_box':            utils.getNumpyAsList(self.bucketBox, 'bucketBox'),
            'approximated_matInside_box':         utils.getNumpyAsList(self.matInsideBox, 'matInsideBox'),
            'approximated_bucket_left_line':      utils.getNumpyAsList(self.bucketLeftLine, 'bucketLeftLine'),
            'approximated_bucket_right_line':     utils.getNumpyAsList(self.bucketRightLine, 'bucketRightLine'),
            'approximated_bucket_mid_line':       utils.getNumpyAsList(self.bucketMidLine, 'bucketMidLine'),
            'bucketWidthPoints':                  utils.getNumpyAsList(self.bucketWidthPointsXCord, 'bucketWidthPointsXCord'),
        }

        if (debug):
            data_dict['debug'] = self.debug_info

        return data_dict




class FMDLAlgo:

    def __init__(self, boxDetectorNetworkPath, roiDelineatorNetworkPath, debugMode=False):
        self.logger = logging.getLogger(__name__)
        self.data = FMDLData(debugMode)

        if self.data.debugMode:
            self.logger.setLevel(logging.DEBUG)

        try:
            self.boxDetector = BoxDetector(boxDetectorNetworkPath)
        except utils.NetworkLoadException:
            self.logger.exception("Could not load the specified boxDetector network,"\
                " you provided:\n%s\n\n", boxDetectorNetworkPath)
            raise utils.NetworkLoadException(boxDetectorNetworkPath)

        try:
            self.roiDelineator = RoiDelineator(roiDelineatorNetworkPath)
        except utils.NetworkLoadException:
            self.logger.exception("Could not load the specified roiDelineator network,"\
                " you provided:\n%s\n\n", roiDelineatorNetworkPath)
            raise utils.NetworkLoadException(roiDelineatorNetworkPath)


    def _load_configs(self, input_config):
        self.data.config = Config(input_config, self.data.expected_configs)
        if not self.data.config.is_valid():
            self.logger.error(
                    'Missing required config items, you are missing:  %s \nYou provided:'\
                    '\n%s\n***Algo will early return***\n\n',
                    self.data.config.missing_configs, input_config)
            return False

        self.logger.info("Equipment config items were successfully read.\n")
        self.logger.debug("The following equipment config items were provided and successfully"\
            " read:\n\n%s\n\n", input_config)
        return True


    def _load_image(self, input_image):
        self.data.load_image(input_image)
        if not self.data.is_valid_image:
            self.logger.error("Could not convert the provided input image from bytearray to"\
                " numpy array.\n***Algo will early return***\n\n")
            return False

        if self.data.debugMode:
            self.data.debug_info.append({'description': 'Input image after conversion from\
             bytearray to numpy','image': visUtils.encodeImageAsBase64(self.data.input_image_np)})
        return True


    def _detect_boundaries(self):
        validResult = self.boxDetector.inferOnSingleImage(self.data)

        self.logger.debug("Result of boxDetector.inferOnSingleImage was: %s\n Highest scoring"\
            " bucket bounding box was\n %s \nwith score of: %s\n Highest scoring matInside bounding"\
            " box was\n%s\nwith score of: %s\n Highest scoring case bounding box was\n%s\nwith"\
            " score of: %s\n\n",validResult, self.data.bucketBoundary,self.data.bucketScore,\
            self.data.matInsideBoundary, self.data.matInsideScore, self.data.caseBoundary,\
             self.data.caseScore)
        
        return validResult
    

    def _find_pixel_conversion_factor(self):
        boxDetectorPredictionValid = utils.getPixel2CmConversionFactor(self.data)

        self.logger.debug("_find_pixel_conversion_factor detected the width of the bucket in pixels to be: %s\n "\
            "pixel2CM_conversion_factor was found to be: %s\nboxDetectorPredictionValid = %s\nbucketWidthPointsXCord = %s\n\n",\
            self.data.pixel2CM_conversion_factor, self.data.bucketWidthPX, boxDetectorPredictionValid, self.data.bucketWidthPointsXCord)

        if not boxDetectorPredictionValid:
            self.logger.warning("Detected bucket boundary NOT valid.\n***Algo will early return***\n")
            return False
        return True


    def _find_best_crop_boundary(self):
        utils.getBestBoundaryToCropWith(self.data)

        self.logger.debug("The best boundary to crop the image with was found to be\n%s\n\n",\
         self.data.bestBoundary)

        if len(self.data.bestBoundary) != 4:
            self.logger.warning("Was unable to find the best boundary to crop the image with"\
                " because NOT ALL of the objects required by config were detected.\n***Algo will"\
                " early return***\n")
            return False
        return True


    def _crop_input_image(self):
        imageCropSuccess = utils.cropImage(self.data)

        if (not imageCropSuccess) or len(self.data.input_image_np_cropped) == 0:
            self.logger.warning("Could not find appropriate bounding boxes to crop the image with."\
                "\n***Algo will early return***\n")
            return False

        self.logger.info("Was able to successfully crop the image using detected bounding boxes.\n")
        if self.data.debugMode:
            self.data.debug_info.append({'description': 'Input image cropped using the best"\
                " detected boundary','image':\
              visUtils.encodeImageAsBase64(self.data.input_image_np_cropped)})
        return imageCropSuccess


    def _detect_roi_uncrop_image(self):
        validRoi = self.roiDelineator.inferOnSingleImage(self.data)

        imageUncropSuccess = utils.uncropImage(self.data)

        if self.data.debugMode:
            self.data.roi_roidDelineatorSize.dtype='uint8'
            self.data.debug_info.append({'description': 'ouput of roiDelineator network in original'\
                ' (raw) size','image': visUtils.encodeImageAsBase64(cv2.cvtColor(\
                self.data.roi_roidDelineatorSize*255, cv2.COLOR_GRAY2BGR))})

            self.data.roi_actualSize.dtype='uint8'
            self.data.debug_info.append({'description': 'ouput of roiDelineator network resized '\
                'to the same scale as input image',
                                         'image': visUtils.encodeImageAsBase64(cv2.cvtColor(\
                                            self.data.roi_actualSize*255, cv2.COLOR_GRAY2BGR))})
            
        return validRoi and imageUncropSuccess


    def _calculate_roi_boundary(self):
        
        if not utils.postProcessRoi(self.data):
            self.logger.warning("_calculate_roi_boundary returned early because of a failure in"\
                " utils.postProcessRoi\n")
            return False
        
        
        if not utils.getRoiContour(self.data):
            self.logger.error("_calculate_roi_boundary returned early because of a failure in"\
                " utils.getRoiContour\n")
            return False
        
        validBoundaryPoints = utils.getRoiBoundaryPoints(self.data)

        if self.data.debugMode:
            self.data.debug_info.append({'description': 'output of roiDelineator, ouput of"\
                " boxDetector, and the approximated roi boundary points all overlayed on the input image',
                                         'image': visUtils.visualizeAllResults(
                                                 self.data.input_image_np,
                                                 self.data.roi_actualSize,
                                                 self.data.bestBoundary,
                                                 self.data.bucketScore,
                                                 self.data.matInsideBoundary,
                                                 self.data.matInsideScore,
                                                 self.data.caseBoundary,
                                                 self.data.caseScore,
                                                 self.data.bucketLeftEdge,
                                                 self.data.bucketRightEdge,
                                                 self.data.bucketMidEdge,
                                                 self.data.approximated_roi_boundary)})
            
        return validBoundaryPoints


    def _validate_returned_dictionary_serializability(self):
        try:
            serializedOutput = json.dumps(self.data.as_dict(self.data.debugMode))
        except:
            self.logger.error("_validate_returned_dictionary_serializability thinks the"\
            " returned dictionary is NOT serializable. Just empty results will be returned.")
            return False

        self.logger.debug("_validate_returned_dictionary_serializability verified that"\
            " returned dictionary is serializable.")
        return True


    def _validate_results(self):
        roiValid = utils.validateApproximatedRoiBoundary(self.data)

        bucketBoxValid = utils.validateBucketBox(self.data)
        matInsideBoxValid = utils.validateMatInsideBox(self.data)
        bucketLinesValid = utils.validateBucketLines(self.data)

        returnedDictionaryIsSerializable = self._validate_returned_dictionary_serializability()

        
        self.logger.debug("The approximated (reduced) roi boundary was valid = %s\n\n", roiValid)

        self.logger.debug("The bucketBox was valid = %s\n\n", bucketBoxValid)

        self.logger.debug("The matInsideBox was valid = %s\n\n", matInsideBoxValid)

        self.logger.debug("The bucketLines were valid = %s\n\n", bucketLinesValid)

        self.logger.debug("returnedDictionaryIsSerializable = %s\n\n", returnedDictionaryIsSerializable)


        if roiValid and bucketBoxValid and matInsideBoxValid and returnedDictionaryIsSerializable:
            self.data.valid = True
            self.logger.info("Successfully calculated and verified final results.\n")
            return True

        else:
            self.data.clearOutputData()
            self.logger.info("Validation of final results failed. We are going to return empty values"\
    " intead of what we calculated because garbage values cause trouble downstream.")
            return False


    def _run_algorithm(self, input_image, input_config):
        #Clear all existing fiels (containing previous execution results) except self.data.debugMode
        self.data.initAllData()
        
        if not self._load_configs(input_config) or not self._load_image(input_image):
            self.logger.error("_run_algorithm returned early because of bad input image or bad configs \n")
            return False

        if not self._detect_boundaries():
            self.logger.warning("_run_algorithm returned early because of a failure in _detect_boundaries\n")
            return False

        if not self._find_best_crop_boundary():
            self.logger.warning("_run_algorithm returned early because of a failure in _find_best_crop_boundary\n")
            return False

        if not self._find_pixel_conversion_factor():
            self.logger.warning("_run_algorithm returned early because of a failure in _find_pixel_conversion_factor\n")
            return False

        if not self._crop_input_image():
            self.logger.warning("_run_algorithm returned early because of a failure in _crop_input_image\n")
            return False

        if not self._detect_roi_uncrop_image():
            self.logger.warning("_run_algorithm returned early because of a failure in _detect_roi_uncrop_image\n")
            return False
            
        if not self._calculate_roi_boundary():
            self.logger.warning("_run_algorithm returned early because of a failure in _calculate_roi_boundary\n")
            return False

        if not self._validate_results():
            self.logger.warning("_run_algorithm returned early because of a failure in _validate_results\n")
            return False

        self.logger.info("_run_algorithm returned the following results (excluding the debug info):\n%s\n\n",
            self.data.as_dict())

        return True


    def execute(self, input_image, input_config):
        achivedFinalResults = self._run_algorithm(input_image, input_config)
        self.logger.info("_run_algorithm achieved final results == %s.\n", achivedFinalResults)

        if not achivedFinalResults:
            self.data.clearOutputData()
        
        return self.data.as_dict(self.data.debugMode)
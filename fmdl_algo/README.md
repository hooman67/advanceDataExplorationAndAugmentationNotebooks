# FMDL_ALGO

## What is this repository for?
* This is a linux-python-tensorflow project, delineating the material inside buckets that are suitable for fragmentation.
* Version: 1.0.0 (Iteration 1)

## How do I get set up?
1. Create a new Anaconda or venv environment with python 3.5 installed.
	* This can be achieved by running:   conda create -n FMDLAlgoEnv python=3.5

2. Install the requirements specified in the requirements.yml file with pip:
	* This can be achieved by running:    pip3 install -r requirements.yml

3. Now you can simply import the FMDLALgo module and use it on your own data. Simply pass in the networks, the input image bytearray, and a config dictionary, as described below.

4. This project has the following dependencies (aside from either Anaconda or Pip3):
	1. python (3.5): pip3 install python=3.5
	2. tensorflow (1.5 or later): pip3 install tensorflow
	3. keras (any version): pip3 install keras
	4. opencv (either 2 or 3): pip3 install opencv-python
	5. numpy (any version): This is required and will be installed by opencv
	5. pillow: (any version): pip3 install pillow
	6. matplotlib (any version): pip3 install matplotlib




## Outside Interface:

### Public Classes:

FMDLAlgo   in FMDLAlgo.py


### Public Methods:

#### 1. __init__(boxDetectorNetworkPath, roiDelineatorNetworkPath, debugMode=False)

##### Input parameters:
	 	1. boxDetectorNetworkPath :
			type:string,
			value: full, normalized path to the box detector network for the equipment type.

	 	2. roiDelineatorNetworkPath :
			type:string,
			value: full, normalized path to the roi delinator network for the equipment type.


##### Throws:
		raise Exception("Could not load the specified network:  " + providedNetworkPath)

		when failing to load one of the specified networks.


##### Return parameters:
		None



#### 2. execute(self, inputImage_bytearray, inputConfigs)

##### Input parameters:
 	1. inputImage_bytearray:
		type: bytearray,
		value: image bytearray, must be grayscale with 3 channels TODO

 	2. inputConfigs :
		type: python dictionary (dict),
		value: Equipment-Specific Config items of the following form: TODO
		shovelConfig = {
    				'measuredBucketWidthCM':              300,
   				 'boxDetectorScoreThresholdBucket':    0.9,
    				'boxDetectorScoreThresholdMatInside': 0.5,
   				 'roiDelineatorScoreThreshold':        0.5,
   				 'minContourArea':                     8000,
   				 'closingKernelSize':                  7,
  				  'closingIterations':                  1,
    				  'erosionKernelSize':                  7,
    				  'erosionIterations':                  1,
    				  'roiBoundaryPointsReductionFactor':   0.01,

      				   'minObjectsRequired':[['bucket'], ['matInside']],
    			}

##### Return Parameters:
		A python dictionary of the following form:

    			results = {
            			'pixel2CM_conversion_factor':  (type: float)
            			'detected_bucketWidth_inPixels': (type: int)
            			'approximated_roi_boundary': (type: numpy.ndarray of the shape 			(numberOfPoints, 2) )
            			'approximated_bucket_box': (type: numpy.ndarray of the shape 			(2, 2) )
            			'approximated_matInside_box': (type: numpy.ndarray of the shape 			(2, 2) )
            			'bucket_valid':          (type: bool)
            			'matInside_valid':       (type: bool)
            			'valid':          (type: bool)
            			}


	**** In debug mode, a �debug� key will be added that contains a list of dictionaries with 		the following format:

    			results = {
            			'pixel2CM_conversion_factor':  (type: float)
            			'detected_bucketWidth_inPixels': (type: int)
            			'approximated_roi_boundary': (type: numpy.ndarray of the shape 			(numberOfPoints, 2) )
            			'approximated_bucket_box': (type: numpy.ndarray of the shape 			(2, 2) )
            			'approximated_matInside_box': (type: numpy.ndarray of the shape 			(2, 2) )
            			'bucket_valid':          (type: bool)
            			'matInside_valid':       (type: bool)
            			'valid':          (type: bool)
			'debug':        [
					{
						'description' : (type: string),
						'image' : (type: base64 string  **Compressed using 										JPEG with Medium quality),
					},
					...
				        ]
            			}


#### Throws:
		None

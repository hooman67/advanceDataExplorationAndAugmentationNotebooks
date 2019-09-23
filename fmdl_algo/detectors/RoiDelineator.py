import numpy as np
import cv2
import logging

import tensorflow as tf

from keras.models import load_model
from keras import backend as K

#TODO: make memory usuage configurable
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 1#0.007 #0.1 for both #0.004 for just unet


class RoiDelineator:
    UNET_INPUT_IMAGE_SIZE = (128,128)

    # Mean IoU metric
    def mean_iou(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)




    def __init__(self, roiDelineatorNetworkPath):
        #This is the actual keras model containing the network
        self.network = load_model(roiDelineatorNetworkPath, custom_objects={'mean_iou':self.mean_iou})


    def inferOnSingleImage(self, data):
        logger = logging.getLogger(__name__)
        
        image_np = data.input_image_np_cropped
        if len(image_np) == 0:
            logger.error("input image provided to RoiDelineators inferOnSingleImage method is empty. We will early return!!!\n")
            return False
        
        try:
            score_threshold = data.config.roiDelineatorScoreThreshold
        except AttributeError:
            logger.error("roiDelineatorScoreThreshold is missing from the parameters supplied to RoiDelineators inferOnSingleImage method. We will early return!!!\n")
            return False
        
        
        try:
            imgUnetSized = cv2.resize(image_np, self.UNET_INPUT_IMAGE_SIZE)
    
            unetOut = self.network.predict(np.expand_dims(imgUnetSized, axis=0), verbose=0)
    
            imUnetOut = (unetOut > score_threshold).astype(np.uint8)
            
            data.roi_roidDelineatorSize = imUnetOut[0, :, :, 0]
        
        except ValueError:
            logger.error("there was something wrong with the croped image input supplied to RoiDelineators inferOnSingleImage method. We will early return!!!\n")
            return False
        
        return len(data.roi_roidDelineatorSize) > 0
import tensorflow as tf
import numpy as np
import logging

INITIAL_NEGATIVE_NUMBER = -1


class BoxDetector:

    def __init__(self, bbDetectorNetworkPath):
        #This is the actual tensorflow graph containing the network
        self.network = tf.Graph()

        with self.network.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(bbDetectorNetworkPath, 'rb') as fid:
                serialized_graph = fid.read()

                od_graph_def.ParseFromString(serialized_graph)

                tf.import_graph_def(od_graph_def, name='')




    def inferOnSingleImage(self, data):
        logger = logging.getLogger(__name__)
        
        image_np = data.input_image_np
        if len(image_np) == 0:
            logger.error("input image provided to boxDetectors inferOnSingleImage method is empty.\
             We will early return!!!\n")
            return False
        
        try:
            score_threshold_bucket = data.config.boxDetectorScoreThresholdBucket
            score_threshold_matInside = data.config.boxDetectorScoreThresholdMatInside
            score_threshold_case = data.config.boxDetectorScoreThresholdCase
 

        except AttributeError:
            logger.error("boxDetectorScoreThresholdBucket or boxDetectorScoreThresholdMatInside is\
             missing from the parameters supplied to boxDetectors inferOnSingleImage method.\
              We will early return!!!\n")
            return False
        
        #TODO: make the session config another parameter.
        with self.network.as_default():

            #HS limiting the GPU usage  config=config_ssd
            with tf.Session() as sess:

                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}

                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores',\
                'detection_classes']:

                    tensor_name = key + ':0'

                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                                tensor_name)


                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                
                try:
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor:\
                     np.expand_dims(image_np, 0)})
                except ValueError:
                    logger.error("there was something wrong with the image input supplied to\
                     boxDetectors inferOnSingleImage method. We will early return!!!\n")
                    return False

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])

                output_dict['detection_classes'] = output_dict['detection_classes']\
                [0].astype(np.uint8)

                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]

                output_dict['detection_scores'] = output_dict['detection_scores'][0]



        logger.debug("The inferOnSingleImage method returned the following output_dict:\n%s\n\n",\
         output_dict)
        
        bucketBoundary, bucketScore,  matInsideBoundary, matInsideScore, caseBoundary, caseScore\
         = self.thresholdPredictions(
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          score_threshold_bucket,
          score_threshold_matInside,
          score_threshold_case)

        
        data.bucketBoundary = bucketBoundary
        data.bucketScore = bucketScore
        data.matInsideBoundary = matInsideBoundary
        data.matInsideScore = matInsideScore
        data.caseBoundary = caseBoundary
        data.caseScore = caseScore


        
        logger.debug("The inferOnSingleImage method added the following to the FMDLData object:\
            \ndata.bucketBoundary = %s\ndata.bucketScore = %s\ndata.matInsideBoundary = \
            %s\ndata.matInsideScore = %s\ndata.caseBoundary = %s\ndata.caseScore = %s\n\n",
                     data.bucketBoundary, data.bucketScore, data.matInsideBoundary,\
                      data.matInsideScore, data.caseBoundary, data.caseScore)


        return True
        


    def thresholdPredictions(self, boxes, classes, scores, score_threshold_bucket,\
     score_threshold_matInside, score_threshold_case):

        global INITIAL_NEGATIVE_NUMBER

        bucketBoundary = tuple()
        bucketScore = INITIAL_NEGATIVE_NUMBER

        matInsideBoundary = tuple()
        matInsideScore = INITIAL_NEGATIVE_NUMBER
        
        caseBoundary = tuple()
        caseScore = INITIAL_NEGATIVE_NUMBER

        for i in range(boxes.shape[0]):
            box = tuple()

            if (classes[i] == 1 and scores[i] >= score_threshold_bucket) \
            or(classes[i] == 2 and scores[i] >= score_threshold_matInside)\
            or(classes[i] == 3 and scores[i] >= score_threshold_case):

                box = tuple(boxes[i].tolist())

                if classes[i]==1 and scores[i] > score_threshold_bucket: #this is a bucket BB
                    if scores[i] > bucketScore:
                        bucketBoundary = box
                        bucketScore = scores[i]

                if classes[i]==2 and scores[i]> score_threshold_matInside: #this is a matInside BB
                    if scores[i] > matInsideScore:
                        matInsideBoundary = box
                        matInsideScore = scores[i]
                        
                        
                if classes[i]==3 and scores[i ]> score_threshold_case: #this is a case BB
                    if scores[i] > caseScore:
                        caseBoundary = box
                        caseScore = scores[i]

        return bucketBoundary, bucketScore, matInsideBoundary, matInsideScore, caseBoundary,\
         caseScore

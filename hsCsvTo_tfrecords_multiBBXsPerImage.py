#hs need to run this inside: /home/hooman/objectDetection/models/research

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys

import os
import os.path
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils.label_map_util import load_labelmap
from object_detection.protos import string_int_label_map_pb2
from collections import namedtuple, OrderedDict




#sys.path.append('/home/hooman/objectDetection/models') # point to your tensorflow dir
#sys.path.append('/home/hooman/objectDetection/models/research/object_detection') # point ot your slim dir
sys.path.append('/home/hooman/objectDetection2/models') # point to your tensorflow dir
sys.path.append('/home/hooman/objectDetection2/models/research/object_detection') # point ot your slim dir

flags = tf.app.flags
flags.DEFINE_string('label_map_path', '/media/hooman/hsSsdPartUbuntu/FM_PROJECT/FMDL_3.1/backhoe/boxDetector_V2_multiclass/try2-withCaseObject-newData/hs_label_map_multiclass_withCase.pbtxt',
                    'Path to label map proto')
flags.DEFINE_string('csv_input', '/media/hooman/hsSsdPartUbuntu/FM_PROJECT/FMDL_3.1/backhoe/boxDetector_V2_multiclass/try2-withCaseObject-newData/trainingSet_shuffled.csv', 'Path to the CSV input')
flags.DEFINE_string('output_path', '/media/hooman/hsSsdPartUbuntu/FM_PROJECT/FMDL_3.1/backhoe/boxDetector_V2_multiclass/try2-withCaseObject-newData/train.record', 'Path to output TFRecord')
FLAGS = flags.FLAGS

path = "/media/hooman/hsSsdPartUbuntu/FM_PROJECT/dataPreparation/FMDL_3.1/backhoe/fmdl-backhoe-trainingData-new_may26-2019/images"#this is the path where your actual .jpg or .png images are





def hsGetDictFromCsv(fullCsvPath):
    csv_file = open(fullCsvPath, "r") 
    data = csv_file.read()
    csv_file.close()

    rows = data.split('\n')

    rowsDict = {}

    for row in rows[1 : len(rows)-1]:
        vals = row.split(',')

        if vals[0] not in rowsDict:
            rowsDict[vals[0]] = []

        rowsDict[vals[0]].append(vals[2:7])
        
    return rowsDict







# TO-DO replace this with label map
def get_label_map_dict(label_map_path):
  """Reads a label map and returns a dictionary of label names to id.
  Args:
    label_map_path: path to label_map.
  Returns:
    A dictionary mapping label names to id.
  """
  label_map = load_labelmap(label_map_path)
  label_map_dict = {}
  for item in label_map.item:
    label_map_dict[item.name] = item.id
  return label_map_dict





def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]




def create_tf_example(group, path, label_map_dict, rowsDict):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    ''' Hs original
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    for box in rowsDict[group.filename]:
        xmins.append(int(box[0]) / width)
        xmaxs.append(int(box[1]) / width)
        ymins.append(int(box[2]) / height)
        ymaxs.append(int(box[3]) / height)
        classes_text.append(box[4].encode('utf8'))
        classes.append(label_map_dict[box[4]])
        

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
    '''

    for box in rowsDict[group.filename]:
        if box[0] != '':
            xmins.append(int(box[0]) / width)
            xmaxs.append(int(box[1]) / width)
            ymins.append(int(box[2]) / height)
            ymaxs.append(int(box[3]) / height)
            classes_text.append(box[4].encode('utf8'))
            classes.append(label_map_dict[box[4]])
        

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example





def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    map_path = os.path.join(os.getcwd(), FLAGS.label_map_path)
    label_map_dict = get_label_map_dict(map_path)
    
    rowsDict = hsGetDictFromCsv(FLAGS.csv_input)
    
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path, label_map_dict, rowsDict)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()

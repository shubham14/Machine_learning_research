from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json.encoder
from glob import glob
from PIL import Image
import pickle

def download_data():
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                        cache_subdir=os.path.abspath('.'),
                                        origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                        extract = True)
        PATH = os.path.dirname(image_zip)+'/train2014/'
    else:
        PATH = os.path.abspath('.')+'/train2014/'
    
    return annotation_file

def prepaer_data(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

        # Store captions and image names in vectors
        all_captions = []
        all_img_name_vector = []

        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            all_img_name_vector.append(full_coco_image_path)
            all_captions.append(caption)

        # Shuffle captions and image_names together
        # Set a random state
        train_captions, img_name_vector = shuffle(all_captions,
                                                all_img_name_vector,
                                                random_state=1)

        # Select the first 30000 captions from the shuffled set
        num_examples = 30000
        train_captions = train_captions[:num_examples]
        img_name_vector = img_name_vector[:num_examples]

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path
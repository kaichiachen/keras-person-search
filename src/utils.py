import xml.etree.ElementTree as ET
import numpy as np
import logging
from src.utils import *
from contextlib import contextmanager
import time
import scipy.misc

@contextmanager
def timer(prefix, index):
    start = time.time()
    yield()
    duration = time.time() - start
    logging.info('%d: %s consume: %fs' % (index, prefix, duration))
    
def save_img(path, image):
    scipy.misc.imsave(path, image)

def generate_bounding_box_from_annotation(annotation, image_shape):
    masks = np.zeros([image_shape[0], image_shape[1]])
    masks[annotation[1]:annotation[3], annotation[0]:annotation[2]] = 1
    return masks

from argparse import ArgumentParser
import tensorflow as tf
import logging
import keras
import pickle
import os
import numpy as np
from keras.applications.vgg16 import VGG16
from src.utils import *
from src.reinforcement import *
from src.metrics import *
from src.features import *
from src.image_helper import *
from PIL import Image

parser = ArgumentParser(description='Test a ReID network.')

parser.add_argument(
    '--gpus', required=False, type=str, default='0',
    help='GPU ID want to use')

parser.add_argument(
    '--max_iters', required=False, type=int, default=100,
    help='Iteration times for testing')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

logging.basicConfig(level=logging.INFO)

with open('data/pid_map_image.txt', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data = u.load()
    
fm_model = VGG16(weights='imagenet')
rl_model = get_q_network('./output/model/model6_epoch.h5')

scale_subregion = float(3)/4
scale_mask = float(1)/(scale_subregion*4)
# Number of steps that the agent does at each image
number_of_steps = 10
# Only search first object
only_first_object = 1
qval = 0
offset = (0, 0)

correct = 0.0
def get_next_minibatch(data):
    num_data = len(data)
    l = []
    inds = np.random.choice(range(num_data), 1, replace=False)
    for i in inds:
        node = np.random.choice(range(len(data[i])), 2)
        l.append([
            [i, data[i][node[0]]],
            [i, data[i][node[1]]]
        ])
    return l


for i in range(args.max_iters):
    #sys.stdout.write('\r'+str(i))
    nodes = get_next_minibatch(data)
    for node in nodes:
        pid = node[0][0]
        target_data = node[0][1]
        search_data = node[1][1]

        target_image = np.array(Image.open(target_data['image']))
        bbox = target_data['boxes'][np.where(target_data['gt_pids']==pid)[0][0]]
        target_image = target_image[bbox[1]:bbox[3],bbox[0]:bbox[2]]

        search_image = np.array(Image.open(search_data['image']))
        region_image = search_image
        iou = 0
        history_vector = np.zeros([24])

        search_iv = get_image_vector(region_image, fm_model)
        target_iv = get_image_vector(target_image, fm_model)
        state = get_state(target_iv, search_iv, history_vector)

        size_mask = (region_image.shape[0], region_image.shape[1])
        original_shape = size_mask

        region_masks = []
        region_mask = np.ones([region_image.shape[0], region_image.shape[1]])
        region_masks.append(region_mask)

        step = 0
        while step < number_of_steps:
            step += 1
            qval = rl_model.predict(state.T, batch_size=1)
            action = (np.argmax(qval))+1

            if action != 6:
                region_mask = np.zeros(original_shape)
                size_mask = (size_mask[0] * scale_subregion, size_mask[1] * scale_subregion)
                region_image, region_mask = do_action(action, region_image, region_mask, offset, size_mask, scale_mask)
                region_masks.append(region_mask)
            else:
                offset = (0, 0)
                if step == 1:
                    absolute_status = 0
                if only_first_object == 1:
                    absolute_status = 0
                image_for_search = mask_image_with_mean_background(region_mask, search_image)
                region_image = image_for_search

                annotation = search_data['boxes'][np.where(search_data['gt_pids']==pid)[0][0]].astype(np.int32)
                gt_mask = generate_bounding_box_from_annotation(annotation, search_image.shape)
                iou = follow_iou(gt_mask, region_mask)
                if iou > 0.5:
                    print('find target!')
                    print('target image: ', target_data['image'])
                    print('bbox: ', bbox[1],bbox[3],bbox[0],bbox[2])
                    print('searchimage: ', search_data['image'])
                    print('iou: ',iou)
                    correct += 1
                break

            history_vector = update_history_vector(history_vector, action)
            search_iv = get_image_vector(region_image, fm_model)
            target_iv = get_image_vector(target_image, fm_model)
            state = get_state(target_iv, search_iv, history_vector)
print(correct/args.max_iters)
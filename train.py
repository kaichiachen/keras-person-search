from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import os
import pickle
from src.model import Model
import logging
import keras

parser = ArgumentParser(description='Train a ReID network.')

parser.add_argument(
    '--gpus', required=False, type=str, default='1',
    help='GPU ID want to use')

parser.add_argument(
    '--max_iters', required=False, type=int, default=500000,
    help='Iteration times for training')

parser.add_argument(
    '--batch_size', required=False, type=int, default=32,
    help='Batch size for training')

parser.add_argument(
    '--featuremap_type', required=False, type=str, default='vgg',
    help='featuremap extractor type(vgg or resnet)')

parser.add_argument(
    '--number_of_steps', required=False, type=int, default=10,
    help='the maximum number of steps allowed for the agent to find person')

parser.add_argument(
    '--log_path', required=False, type=str, default='./log',
    help='path to store tensorboard log event')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

logging.basicConfig(level=logging.INFO)

with open('data/pid_map_image_update.txt', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    data = u.load()
    
ct = Model(data, batch_size=args.batch_size)
ct.load_feature_map_model(args.featuremap_type)
ct.train_model(max_iters=args.max_iters, number_of_steps=args.number_of_steps, log_path=args.log_path)
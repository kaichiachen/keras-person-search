from src.reinforcement import *
from src.features import *
from src.metrics import *
from src.image_helper import *
from src.utils import *
from keras.applications.vgg19 import VGG19
from keras.models import Model as kerasModel
from keras.callbacks import TensorBoard
import keras
import tensorflow as tf
from PIL import Image
import random
import sys
from contextlib import contextmanager
import logging
from src.replay_buffer import PrioritizedReplayBuffer
from common.segment_tree import SumSegmentTree, MinSegmentTree
from common.schedules import LinearSchedule

class Model(object):

    def __init__(self, data, pretrained_model='0', batch_size=32):
        self.data = data
        self.num_data = len(self.data)
        self.batch_size = batch_size
        self.pretrained_model=pretrained_model
        
        self.scale_subregion = float(3)/4
        self.scale_mask = float(1)/(self.scale_subregion*4)
        self.buffer_experience_replay = 50000
        self.gamma = 0.90
        self.replay = PrioritizedReplayBuffer(self.buffer_experience_replay, alpha=0.6)
            
        
    def load_feature_map_model(self, t):
        if t == 'vgg':
            self.feature_map_extractor_model = VGG19(weights='imagenet')#, include_top=False, input_shape=(800,800,3))
        else:
            self.feature_map_extractor_model = VGG19(weights='imagenet')#, include_top=False, input_shape=(800,800,3))
            
        self.feature_map_extractor_model = kerasModel(input=self.feature_map_extractor_model.input, output=self.feature_map_extractor_model.get_layer('fc2').output)
        self.feature_map_extractor_model.summary()
        self.rl_model = get_q_network(self.pretrained_model)
        
    def get_next_node(self):
        inds = np.random.choice(range(self.num_data), 1, replace=False)[0]
        node = np.random.choice(range(len(self.data[inds])), 2)
        
        return inds, self.data[inds][node[0]], self.data[inds][node[1]], False
        #return 1701, self.data[1701][7], self.data[1701][8], False
    
    def save_model(self, epoch):
        print('\nsave model')
        string = './output/model/model' + str(int(epoch)) + '_epoch' + '.h5'
        self.rl_model.save_weights(string, overwrite=True)
        
    def train_iterator(self, debug, i, number_of_steps, epsilon, start_train_iters=10):
        pid, target_data, search_data, not_contain = self.get_next_node()

        target_image = np.array(Image.open(target_data['image']))
        bbox = target_data['boxes'][np.where(target_data['gt_pids']==pid)[0][0]]
        target_image = target_image[bbox[1]:bbox[3],bbox[0]:bbox[2]]

        search_image = np.array(Image.open(search_data['image']))
        annotation = search_data['boxes'][np.where(search_data['gt_pids']==pid)[0][0]]#.astype(np.int32)
        gt_mask = generate_bounding_box_from_annotation(annotation, search_image.shape)
        region_mask = np.ones([search_image.shape[0], search_image.shape[1]])
        shape_gt_mask = np.shape(gt_mask)

        step = 0
        new_iou = 0
        last_matrix = 0
        region_image_from_original = region_image = search_image
        offset = (0, 0)
        size_mask = (search_image.shape[0], search_image.shape[1])
        original_shape = size_mask
        old_region_mask = region_mask
        region_mask = np.ones([search_image.shape[0], search_image.shape[1]])

        iou = follow_iou(gt_mask, region_mask)
        # init of the history vector that indicates past actions (7 actions * 4 steps in the memory)
        history_vector = np.zeros([24])
        # computation of the initial state
        if target_image.shape[0] <=1 or target_image.shape[1] <=1:
            print('pid:',pid,' img_path:',target_data["image"], 'shape:', target_image.shape, 'anno:', np.where(target_data['gt_pids']==pid), 'pids', target_data['gt_pids'], 'boxes', target_data['boxes'], 'box',target_data['boxes'][np.where(target_data['gt_pids']==pid)[0][0]])
            return 0
        search_iv = get_image_vector(region_image, self.feature_map_extractor_model)
        target_iv = get_image_vector(target_image, self.feature_map_extractor_model)
        state = get_state(target_iv, search_iv, history_vector)
        action = 0
        reward = 0
        
        region_masks = []
        history_callback = keras.callbacks.History()
        while step < number_of_steps:
            step += 1
            qval = self.rl_model.predict(state.T, batch_size=1)

            # we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
            #if new_iou > 0.5:
            #    action = 6
            # epsilon-greedy policy
            if random.random() < epsilon:
                action = np.random.randint(1, 7)
            else:
                action = (np.argmax(qval))+1

            # terminal action
            if action == 6:
                new_iou = follow_iou(gt_mask, region_mask)
                reward = get_reward_trigger(new_iou)
                if new_iou > 0.5 and not debug:
                    image = mask_image_with_mean_background(gt_mask, search_image, [0,255,0])
                    image = np.concatenate([image] + [mask_image_with_mean_background(region_mask, search_image, [255,0,0]) for region_mask in region_masks],axis=1)
                    #image = mask_image_with_mean_background(region_mask, search_image)
                    save_img('./output/img/%d-%d-%f-%s.jpg' % (i,pid,new_iou,search_data['image'].split('/')[-1][:-4]), image)
            elif action == 7:
                if not_contain: reward = 1
                else: reward = -1
            elif step == number_of_steps and action != 6:
                new_iou = follow_iou(gt_mask, region_mask)
                if new_iou > 0.5:
                    reward = -0.5
                else:
                    reward = 1
            else:
                region_mask = np.zeros(original_shape)
                size_mask = (size_mask[0] * self.scale_subregion, size_mask[1] * self.scale_subregion)
                region_image_from_original, region_image, region_mask = do_action(action, search_image, region_image, region_mask, offset, size_mask, self.scale_mask)
                region_masks.append(region_mask)
                new_iou = follow_iou(gt_mask, region_mask)
                reward = get_reward_movement(iou, new_iou)
                iou = new_iou

            history_vector = update_history_vector(history_vector, action)
            with timer('update_new_state', i):
                search_iv = get_image_vector(region_image_from_original, self.feature_map_extractor_model)
                new_state = get_state(target_iv, search_iv, history_vector)
            # Experience replay storage
            if len(self.replay) < self.buffer_experience_replay:
                self.replay.add(state, action, reward, new_state, float(action>=6))
                #self.replay.append((state, action, reward, new_state))
                
            losses = []
            if i >= start_train_iters:
                self.replay.add(state, action, reward, new_state, float(action>=6))
                experience = self.replay.sample(self.batch_size, beta=self.beta_schedule.value(i))
                (obs, actions, rewards, new_obs, dones, weights, batch_idxes) = experience
                
                # we pick from the replay memory a sampled minibatch and generate the training samples
                for memory in zip(obs, actions, rewards, new_obs):
                    X_train = []
                    y_train = []
                    old_state, action, reward, new_state = memory
                    old_qval = self.rl_model.predict(old_state.T, batch_size=1)
                    newQ = self.rl_model.predict(new_state.T, batch_size=1)
                    maxQ = np.max(newQ)
                    y = old_qval.T
                    if action != 6: #non-terminal state
                        update = (reward + (self.gamma * maxQ))
                    else: #terminal state
                        update = reward
                    y[action-1] = update #target output
                    X_train.append(old_state)
                    y_train.append(y)
                    X_train = np.array(X_train).astype("float32")[:, :, 0]
                    y_train = np.array(y_train).astype("float32")[:, :, 0]

                    self.rl_model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=0, callbacks=[history_callback])
                    losses.append(history_callback.history['loss'][0])
                td_errors = losses
                new_priorities = np.abs(td_errors) + 1e-6
                self.replay.update_priorities(batch_idxes, new_priorities)
                
                state = new_state

            if action == 6: break
        if i >= start_train_iters: losses = history_callback.history['loss']
        return (reward, step, losses)

    def train_model(self, max_iters=100000, number_of_steps=10, log_path='./log', debug=False):
        tb = TensorBoard(log_path)
        tb.set_model(self.rl_model)
        
        self.beta_schedule = LinearSchedule(100000,initial_p=0.4,final_p=1.0)
        epsilon = 1.
        for i in range(max_iters):
            if epsilon > 0.1:
                epsilon *= 0.98
            with timer('all', i):
                reward, step, loss = self.train_iterator(debug, i, number_of_steps, epsilon)
            
            if not debug:
                summary = tf.Summary()
                reward_value = summary.value.add()
                reward_value.simple_value = reward
                reward_value.tag = 'reward'
                step_value = summary.value.add()
                step_value.simple_value = step
                step_value.tag = 'step'
                if len(loss) > 0:
                    loss_value = summary.value.add()
                    loss_value.simple_value = sum(loss)/len(loss)
                    loss_value.tag = 'loss'
                tb.writer.add_summary(summary, i)
                tb.writer.flush()
            #sys.stdout.write('\r'+str(i))
            if (i+1)%10000 is 0:
                self.save_model(i/10000+1)

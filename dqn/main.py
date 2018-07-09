"""
    modified from 
    https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
    tf implementation of dueling q network 
    https://arxiv.org/pdf/1511.06581.pdf 
    NOTE: from the op, there are two points that is different to the paper, 
    1. the paper dueling dqn is not split, they use the same fc, in the qnetwork.py we split
    2. the target graph do not get updated everystep in the original paper, here we update the target network a little each step
"""

from __future__ import division
import gym
import gym.spaces
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from qnetwork import Qnetwork
from experience_buffer import ExperienceBuffer

BATCH_SIZE = 32
UPDATE_FREQ = 4
GAMMA = 0.99 # discount
START_EP = 1 # ep-greedy starting
END_EP = 0.1 # ep-greedy end
REDUCE_EP_STEPS = 10000 # num steps to reduce start_ep to end_ep
NUM_EPISODES = 100000 # num game episodes we play
PRE_TRAIN_STEPS = 500 # num steps of random actions before training
MAX_STEPS = 50 # num of steps in each of our episode
LOAD_MODEL = False # load a model
PATH = "./ddqn" # checkpoint path to save our model
FCL_SIZE = 512 # final fc layer before dueling dqn
TAU = 0.001 # rate to update our target network


def update_target_graph(vars, tau):
    """
        NOTE: Normally the target network is fixed without updating after a number of steps
        'modified version of the update where the target network is updated at every step, 
        but only a small amount (determined by a variable tau. 
        This variable is typically set to something like 0.001, 
        such that 1000 updates is roughly the equivalent of updating the network fully every 1000 steps. 
        By smoothly updating the target network it prevents potential instability that can arise from a sudden large change in the target network.'

        params:
            vars - tensorflow variables to update
            tau - hyper-parameter
    """
    total_vars = len(vars)
    op_holder = []
    for idx, var in enumerate(vars[0:total_vars//2]):
        # assign values from the tf variables
        new_value_op = (var.value() * tau) + ((1-tau)* vars[idx+total_vars//2].value())
        update_value_op= vars[idx+total_vars // 2].assign(new_value_op)
        op_holder.append(update_value_op)
    return op_holder

def update_target(sess, op_holder):
    for op in op_holder:
        sess.run(op)

def init_tf_variables():
    global_ops = tf.global_variables_initializer()
    saver = tf.train.Saver()
    trainables = tf.trainable_variables()
    target_ops = update_target_graph(trainables, TAU)
    return global_ops, saver, target_ops

def check_or_create_dirs():
    if not os.path.exists(PATH):
        os.makedirs(PATH)

def ep_greedy(sess, ep, total_steps, main_q_network, observation):
    if np.random.rand(1) < ep or total_steps < PRE_TRAIN_STEPS:
        action = np.random.randint(0, 3)
    else:
        prediction = sess.run(main_q_network.predict, feed_dict={main_q_network.image_inputs: [observation]})
        action = prediction[0]
        #print("action from network %d" % action)

    return action

def decrease_ep(total_steps, ep, step_decrease):
    if total_steps > PRE_TRAIN_STEPS:
        if ep > END_EP:
            ep -= step_decrease
    return ep

def main():

    global UPDATE_FREQ

    tf.reset_default_graph()
    
    env = gym.make('Freeway-v0')

    num_actions = env.action_space.n
    env_shape = env.observation_space.shape
    image_shape = [env_shape[0], env_shape[1], env_shape[2]]

    # init our double dqn
    main_q_network = Qnetwork(image_shape, FCL_SIZE, num_actions)
    target_q_network = Qnetwork(image_shape, FCL_SIZE, num_actions)

    # check ckpt dirs
    check_or_create_dirs()
    
    glbal_ops, saver, target_ops = init_tf_variables()

    ep = START_EP
    step_decrease = (START_EP - END_EP) / REDUCE_EP_STEPS
    
    ex_buffer = ExperienceBuffer()

    j_list = []
    r_list = []

    total_steps = 0


    with tf.Session() as sess:
        sess.run(glbal_ops)

        if LOAD_MODEL:
            print('LOADING MODEL ... ')
            ckpt = tf.train.get_checkpoint_state(PATH)
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        for episode in range(NUM_EPISODES):
            episode_buffer = ExperienceBuffer()
            observation = env.reset()
            done = False
            reward_all = 0
            j = 0

            while j < MAX_STEPS:
                j+=1
                env.render()
                action = ep_greedy(sess, ep, total_steps, main_q_network, observation)

                new_observation, reward, done, info = env.step(action)

                total_steps += 1
                episode_results = np.array([observation, action, reward, new_observation, done])
                episode_experience = np.reshape(episode_results, [1, 5])
                episode_buffer.add(episode_experience)
                #print("accumulated episdoes experience %d" % len(episode_buffer.buffer))
                ep = decrease_ep(total_steps, ep, step_decrease)
                #print("epsilon %f" % ep)
                #print(total_steps)
                if total_steps > PRE_TRAIN_STEPS:
                    if total_steps % (UPDATE_FREQ) == 0:
                        train_batch = ex_buffer.sample(BATCH_SIZE)
                        # take our new observation to perform the double-dqn
                        q1 = sess.run(main_q_network.predict, feed_dict={main_q_network.image_inputs: np.stack(train_batch[:,3])})
                        q2 = sess.run(target_q_network.q_output, feed_dict={target_q_network.image_inputs: np.stack(train_batch[:,3])})

                        end_multiplier = - (train_batch[:, 4] - 1)
                        # NOTE: choose q_values from target q network using main q network
                        double_q = q2[range(BATCH_SIZE), q1]
                        target_q = train_batch[:,2] + (GAMMA * double_q * end_multiplier)
                        # NOTE: update our main q network after calculate q target from double q
                        _ = sess.run(main_q_network.train_op, feed_dict={main_q_network.image_inputs: np.stack(train_batch[:,0]), main_q_network.target_q: target_q, main_q_network.actions: train_batch[:, 1]})
                        
                        # NOTE: we update our target network separately
                        update_target(sess, target_ops)

                reward_all += reward
                observation = new_observation

                if done == True:
                    break

            ex_buffer.add(episode_buffer.buffer)
            j_list.append(j)
            r_list.append(reward_all)

            if episode % 1000 == 0 :
                saver.save(sess, PATH+'/model-'+str(episode)+'.ckpt')
                print("saved model at %d" % episode)
            
            if len(r_list) % 10 == 0:
                print(total_steps, np.mean(r_list[-10:]), ep)
        
        saver.save(sess, PATH+'/model-'+str(episode)+'.ckpt')
    print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

if __name__ == '__main__':
    main()
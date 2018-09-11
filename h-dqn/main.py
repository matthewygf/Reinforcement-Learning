import os
import argparse
import numpy as np

from utils.monitor import Monitor
from utils.schedules import LinearSchedule
from envs.ale_atari_env import AtariEnv
from envs.agent import Critic
import envs.env_wrapper as wrapper

from deepq.replay_buffer import ReplayBuffer
from deepq.models import Q_network
from deepq.controller import Controller, MetaController

# computer vision packages 
from PIL import Image, ImageDraw

# tensorflow stuffs
import tensorflow as tf
from tensorflow import logging
from utils.tensorboard import TensorboardVisualizer
from utils import tf_util as U

FLAGS = tf.app.flags.FLAGS
D2_MEMORY_SIZE = 5e4
D1_MEMORY_SIZE = 1e6
LEARNING_RATE = 5e-4
MAX_EPISODE = 9999999
EXPLORATION_FRACTION = 0.1
EXPLORATION_FINAL_EPS = 0.02

# in the paper, warm up steps are for lower level controller 
# to learn how to solve a particular goal
# effectively pretrained.
WARMUP_STEPS = 2.5e6
# make the training more stable
UPDATE_TARGET_NETWORK_FREQ = 1000

def main(_):
    # create visualizer
    visualizer = TensorboardVisualizer()
    monitor = Monitor(FLAGS)
    log_dir = monitor.log_dir
    visualizer.initialize(log_dir, None)
    
    # initialize env
    atari_env = AtariEnv(monitor)
    #screen_shot_subgoal(atari_env)

    # we should probably follow deepmind style env
    # stack 4 frames and scale float
    env = wrapper.wrap_deepmind(atari_env, frame_stack=True, scale=True)

    # get default tf_session
    sess = U.get_session()

    # create q networks for controller
    controller_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    controller_network = Q_network(env.observation_space, env.action_space.n, controller_optimizer, scope='controller')
    controller = Controller(controller_network, env.action_space.n)
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(EXPLORATION_FRACTION * monitor.num_timesteps),
                                 initial_p=1.0,
                                 final_p=EXPLORATION_FINAL_EPS)
    # create q networks for meta-controller
    num_goals = env.unwrapped.goals_space.n
    metacontroller_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    metacontroller_network = Q_network(env.observation_space, num_goals, metacontroller_optimizer, scope='metacontroller')
    metacontroller = MetaController(metacontroller_network, num_goals)
    # Create the schedule for exploration starting from 1.
    exploration2 = LinearSchedule(schedule_timesteps=int(EXPLORATION_FRACTION * monitor.num_timesteps),
                                 initial_p=1.0,
                                 final_p=EXPLORATION_FINAL_EPS)
    # initialize experience replay
    controller_replay_buffer = ReplayBuffer(D1_MEMORY_SIZE)
    metacontroller_replay_buffer = ReplayBuffer(D2_MEMORY_SIZE)
    
    total_extrinsic_reward = 0
    total_intrinsic_reward = 0
    total_episode = 0
    total_steps = 0
    ep = 0
    init_ob = env.reset()

    U.initialize()
    # initialize target network in both controller and meta
    sess.run(metacontroller.network.update_target_op)
    sess.run(controller.network.update_target_op)

    #TODO: intrinsic critic
    while ep < 1: # count number of steps 
        # init environment game play variables
        extrinsic_reward = []
        intrinsic_reward = []
        step = 0

        # load ckpt if presence 
        model_path = tf.train.latest_checkpoint(monitor.ckpt_dir)
        if model_path is not None:
            U.load_variables(model_path)
            logging.info('loaded model from %s' % model_path)
        
        init_ob = env.reset()

        observation = np.reshape(init_ob['observation'], (1, )+init_ob['observation'].shape)
        desired_goal = metacontroller.sample_act(sess, observation, update_eps=1.0)[0]
        
        env.unwrapped.desired_goal = desired_goal

        # given predicted goal, we encode this goal bounding mask to the observation np array
        ob_with_g = env.unwrapped._add_goal_mask(init_ob['observation'], desired_goal)

        # NOTE: Below code verify added mask correctly
        # for i in range(ob_with_g.shape[-1]):
        #     ob = ob_with_g[:,:,i]
        #     image = Image.fromarray(ob)
        #     image = image.convert('RGB')
        #     image.save('test_%i.png' % i)

        done = False
        reached_goal = False

        while not done or reached_goal:
            update_eps1 = exploration.value(step)
            ob_with_g = np.reshape(ob_with_g, (1, )+ob_with_g.shape)
            primitive_action_t = controller.sample_act(sess, ob_with_g, update_eps=update_eps1)
            # obtain extrinsic reward from environment
            ob_tp1, extrinsic_reward_t, done_t, info = env.step(primitive_action_t)
            reached_goal = env.unwrapped.reached_goal(desired_goal)
            ob_with_g_tp1 = env.unwrapped._add_goal_mask(ob_tp1['observation'], desired_goal)
            # obtain intrinsic reward from critic
            critic = Critic(env.unwrapped)
            intrinsic_reward_t = critic.criticize(desired_goal, reached_goal, primitive_action_t, done_t)
            controller_replay_buffer.add(ob_with_g, primitive_action_t, intrinsic_reward_t, ob_with_g_tp1, done_t)
            
            # TODO: sample from replay_buffer1 to train controller
            # train controller.

            # TODO: join train meta-controller only sample from replay_buffer2 to train meta-controller
            if step >= WARMUP_STEPS:
                pass

            if step % UPDATE_TARGET_NETWORK_FREQ == 0:
                sess.run(controller.network.update_target_op)
                # its fine, we aren't really training meta dqn until after.
                sess.run(metacontroller.network.update_target_op)

            ob_with_g = ob_with_g_tp1
            done = done_t
            step += 1

        ep += 1
        

def screen_shot_subgoal(env, use_small=False):
    """
    params:
        env - AtariEnv wrapper, gym.GoalEnv specifically
    """
    init_screen = env._get_image(show_goals=True, use_small=use_small)
    image = Image.fromarray(init_screen, mode='RGB')
    if use_small:
        image.save('subgoal_boxes_small.png')
    else:
        image.save('subgoal_boxes.png')
    logging.info('screenshoted image with goals')    


if __name__ == '__main__':
    tf.app.run()
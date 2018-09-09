import os
import argparse
import numpy as np

from utils.monitor import Monitor
from envs.ale_atari_env import AtariEnv
import envs.env_wrapper as wrapper

from deepq.replay_buffer import ReplayBuffer
from deepq.models import Q_network
from deepq.controller import Controller, MetaController
# computer vision packages 
from PIL import Image, ImageDraw

# tensorflow stuffs
import tensorflow as tf
from utils.tensorboard import TensorboardVisualizer
from utils import tf_util as U

FLAGS = tf.app.flags.FLAGS
D2_MEMORY_SIZE = 5e4
D1_MEMORY_SIZE = 1e6
LEARNING_RATE = 5e-4
MAX_EPISODE = 99999

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

    # create tf_session
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    sess = U.get_session(config=tf_config)

    # create q networks for controller
    controller_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    controller_network = Q_network(env.observation_space, env.action_space.n, controller_optimizer, scope='controller')
    controller = Controller(controller_network, env.action_space.n)

    # create q networks for meta-controller
    num_goals = env.unwrapped.goals_space.n
    metacontroller_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    metacontroller_network = Q_network(env.observation_space, num_goals, metacontroller_optimizer, scope='metacontroller')
    metacontroller = MetaController(metacontroller_network, num_goals)

    # initialize experience replay
    controller_replay_buffer = ReplayBuffer(D1_MEMORY_SIZE)
    metacontroller_replay_buffer = ReplayBuffer(D2_MEMORY_SIZE)
    
    extrinsic_reward = 0
    intrinsic_reward = 0
    total_episode = 0
    total_steps = 0

    #TODO: actual training, and intrinsic critic

def screen_shot_subgoal(env, use_small=False):
    """
    params:
        env - AtariEnv wrapper, gym.GoalEnv specifically
    """
    init_screen = env._get_image(show_goals=True, use_small=use_small) # 210, 160, 3
    image = Image.fromarray(init_screen, mode='RGB')
    if use_small:
        image.save('subgoal_boxes_small.png')
    else:
        image.save('subgoal_boxes.png')


if __name__ == '__main__':
    tf.app.run()
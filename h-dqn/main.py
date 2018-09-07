import os
import argparse
import numpy as np

from utils.monitor import Monitor
from envs.ale_atari_env import AtariEnv
import envs.env_wrapper as wrapper

from deepq.replay_buffer import ReplayBuffer
from deepq.models import Q_network
from deepq.controller import Controller
# computer vision packages 
from PIL import Image, ImageDraw

# tensorflow stuffs
import tensorflow as tf
from utils.tensorboard import TensorboardVisualizer

FLAGS = tf.app.flags.FLAGS
EXP_MEMORY_SIZE = 50000
LEARNING_RATE = 5e-4

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

    # we have meta-controller , controller for each goal
    num_goals = env.unwrapped.goals_space.n

    # create q networks for each goal controller
    hdq_optimizers = []
    goal_controllers = []
    for i in range(num_goals):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        network = Q_network(env.observation_space, env.action_space, optimizer, scope='goal%i' % i)
        # create goal controller for each goal
        controller = Controller(network, 'goal%i' % i, env.observation_space)
        goal_controllers.append(controller)
        hdq_optimizers.append(optimizer)

    # TODO: finish controller implementation ie: pick actions, learn, update.


    replay_buffer = ReplayBuffer(EXP_MEMORY_SIZE)
    # initialize experience replay

    # initialize agents

    # count rewards, states, actions, goals

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
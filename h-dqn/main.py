import os
import argparse
import numpy as np

from utils.monitor import Monitor
from envs.ale_atari_env import AtariEnv
import envs.env_wrapper as wrapper

from deepq.replay_buffer import ReplayBuffer

# computer vision packages 
from PIL import Image, ImageDraw

# tensorflow stuffs
import tensorflow as tf
from utils.tensorboard import TensorboardVisualizer
FLAGS = tf.app.flags.FLAGS

EXP_MEMORY_SIZE = 50000

def main(_):
    # create visualizer
    visualizer = TensorboardVisualizer()
    monitor = Monitor(FLAGS)
    log_dir = monitor.log_dir
    visualizer.initialize(log_dir, None)

    # initialize env
    atari_env = AtariEnv(monitor)
    # screen_shot_subgoal(atari_env)

    replay_buffer = ReplayBuffer(EXP_MEMORY_SIZE)

    # we should probably follow deepmind style env
    # stack 4 frames and scale float
    env = wrapper.wrap_deepmind(atari_env, frame_stack=True, scale=True)
    # TODO: verified whether to use _SMALL
    # env.compute_reward(0,0, None)

    # create q networks for meta controller, sub controller

    # initialize experience replay

    # initialize agents

    # count rewards, states, actions, goals

def screen_shot_subgoal(env):
    """
    params:
        env - AtariEnv wrapper, gym.GoalEnv specifically
    """
    init_screen = env._get_image(show_goals=True) # 210, 160, 3
    image = Image.fromarray(init_screen, mode='RGB')
    image.save('subgoal_boxes.png')


if __name__ == '__main__':
    tf.app.run()
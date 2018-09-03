import os
import argparse
import numpy as np
import tensorflow as tf
from utils.tensorboard import TensorboardVisualizer
from utils.monitor import Monitor
from envs.env_with_goals import EnvWithGoals
import envs.env_wrapper as wrapper
# computer vision
from PIL import Image, ImageDraw

FLAGS = tf.app.flags.FLAGS

# GOALS DEFINED
# TODO: how do we spot goals ?! 
# in 210, 160 pixels 
LOWER_RIGHT_LADDER = [(130, 142), (140, 148)]
KEY = [(12, 100), (24, 120)]
RIGHT_DOOR = [(130, 50), (144, 80)]
# in 84, 84 pixels ?
# from https://github.com/hoangminhle/hierarchical_IL_RL.git
LOWER_RIGHT_LADDER_SMALL = [(69, 68), (73, 71)]
KEY_SMALL = [(7, 41), (11, 45)]
RIGHT_DOOR_SMALL = [(70, 20), (73, 35)]

def main(_):
    # create visualizer
    visualizer = TensorboardVisualizer()
    monitor = Monitor(FLAGS)
    log_dir = monitor.log_dir
    visualizer.initialize(log_dir, None)

    # initialize env
    atari_env = EnvWithGoals(monitor)
    # we should probably follow deepmind style env
    env = wrapper.wrap_deepmind(atari_env, frame_stack=True, scale=True)
    
    print (env.observation_space)

    # set up our goals inside the environment 
    # TODO: verified whether to use _SMALL
    
    # screenshot our goals
    # goals_set_large = [LOWER_RIGHT_LADDER, KEY, LOWER_RIGHT_LADDER, RIGHT_DOOR]
    # screen_shot_subgoal(atari_env, goals_set_large)

    # create q networks for meta controller, sub controller

    # initialize experience replay

    # initialize agents

    # count rewards, states, actions, goals

def screen_shot_subgoal(env, goals, multiplier=1):
    """
    params:
        env - atari_env wrapper
        goals - list of goal coordinate in [ (x1,y1), (x2,y2) ]
        multiplier - just to quickly scale the coordinate
    """
    init_screen = env.ale.getScreenRGB() # 210, 160, 3
    image = Image.fromarray(init_screen, mode='RGB')
    image.save('init_screen.png')
    image = Image.open('init_screen.png')
    draw = ImageDraw.Draw(image)
    for i in range(len(goals)):
        coordinates = []
        for j in range(len(goals[i])):
            coordinates.append(tuple(multiplier * x for x in goals[i][j]))
        draw.rectangle(coordinates, outline='white')
    image.save('subgoal_boxes.png')


if __name__ == '__main__':
    tf.app.run()
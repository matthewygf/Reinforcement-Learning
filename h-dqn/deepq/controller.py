import tensorflow as tf
from gym import spaces

class Controller(object):
    def __init__(self, network, scope, observation_space):
        with tf.variable_scope(scope=scope+'/controller', reuse=tf.AUTO_REUSE):
            if isinstance(observation_space, spaces.Dict):
                ob = observation_space.spaces['observation']
            else:
                ob = observation_space

            self.obs_t_input = tf.placeholder(tf.float32, shape=(None, )+ob.shape, name='obs_t')

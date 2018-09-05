import tensorflow as tf
from gym import spaces

class Q_network(object):
    def __init__(self, observation_space, double = True, dueling = True, reuse=None, data_format='channels_last'):
        active_fn = tf.nn.relu
        with tf.variable_scope('q_networks', reuse=reuse):
            # set up placeholder 
            # observation should be scaled already
            if isinstance(observation_space, spaces.Dict):
                ob = observation_space.spaces['observation']
            else:
                ob = observation_space

            self.obs_t_input = tf.placeholder(tf.float32, shape=(None, )+ob.shape, name='obs_t')
            self.action_t = tf.placeholder(tf.int32, shape=[None], name='action')
            self.reward_t = tf.placeholder(tf.float32, shape=[None], name='reward')
            self.obs_tp1_input = tf.placeholder(tf.float32, shape=(None, )+ob.shape, name='obs_tp1')
            self.done = tf.placeholder(tf.float32, shape=[None], name='done')
            self.importance_weight = tf.placeholder(tf.float32, shape=[None], name='weight')

            # build first q 
            with tf.variable_scope('online_q'):
                c1 = tf.layers.conv2d(self.obs_t_input, 32, 8, 4, activation=active_fn, data_format=data_format)            
                c2 = tf.layers.conv2d(c1, 64, 4, 2, activation=active_fn, data_format=data_format)
                c3 = tf.layers.cov2d(c2, 64, 3, 1, activation=active_fn, data_format=data_format)
                # TODO:
import tensorflow as tf
from gym import spaces

# TODO: ideally make them use same implementation, but it's just clearer LOL

class MetaController(object):
    def __init__(self, network, num_goals):
        self.network = network

        with tf.variable_scope('metacontroller', reuse=tf.AUTO_REUSE):
            # epsilon greedy
            self.stochastic_t = tf.placeholder(tf.bool, shape=(), name='stochastic_t')
            # anneal epsilon
            self.update_eps_t = tf.placeholder(tf.float32, shape=(), name='update_eps_t')
            # learn step
            self.epsilon = tf.get_variable('eps', shape=(), initializer=tf.constant_initializer(value=0))

            # max of our online network
            # when getting this op, we should remember to feed observation to online network.
            deterministic_goals = tf.argmax(network.q_out, axis=1)

            batch_size = tf.shape(network.obs_t_input)[0]

            random_goals = tf.random_uniform(shape=([batch_size]), 
                                              minval=0, 
                                              maxval=num_goals, 
                                              dtype=tf.int64)
            
            should_choose_random = (tf.random_uniform(tf.stack([batch_size]), 
                                                      minval=0, 
                                                      maxval=1, 
                                                      dtype=tf.float32) < self.epsilon)

            stochastic_goals = tf.where(should_choose_random, random_goals, deterministic_goals)

            self.output_goals = tf.cond(self.stochastic_t, 
                                     lambda: stochastic_goals, 
                                     lambda: deterministic_goals) 
            
            self.updates_eps_op = self.epsilon.assign(tf.cond(self.update_eps_t >= 0, 
                                                              lambda: self.update_eps_t, 
                                                              lambda: self.epsilon))
    
    def sample_act(self, sess, observations, stochastic=True, update_eps = -1):
        goal, _ = sess.run([self.output_goals, self.updates_eps_op], 
                             feed_dict={self.network.obs_t_input: observations,
                                        self.stochastic_t: stochastic,
                                        self.update_eps_t: update_eps})
        return goal

    def get_q(self, sess, observation):
        q = sess.run([self.network.q_out],
                     feed_dict={self.network.obs_t_input: observation})
        return q

    def train(self, sess, observation_t, goal_t, extrinsic_reward_t, observation_tp1, done_t, importance_weights, online_q_tp1):
        td_error, _ = sess.run([self.network.td_error, self.network.optimize_op], 
                               feed_dict={self.network.obs_t_input: observation_t,
                                          self.network.action_t: goal_t,
                                          self.network.reward_t: extrinsic_reward_t,
                                          self.network.obs_tp1_input: observation_tp1,
                                          self.network.importance_weights: importance_weights,
                                          self.network.done_t: done_t,
                                          self.network.online_q_tp1: online_q_tp1})
        return td_error


class Controller(object):
    """
    solve the predicted goal (encoded into the observation), by maximizing the intrinsic reward.
    hence our network here should use intrinsic rewards only.
    """
    def __init__(self, network, num_actions):
        self.network = network

        with tf.variable_scope('controller', reuse=tf.AUTO_REUSE):
            self.stochastic_t = tf.placeholder(tf.bool, shape=(), name='stochastic_t')
            # anneal epsilon
            self.update_eps_t = tf.placeholder(tf.float32, shape=(), name='update_eps_t')
            # learn step
            self.epsilon = tf.get_variable('eps', shape=(), initializer=tf.constant_initializer(value=0))

            # max of our online network
            # when getting this op, we should remember to feed observation to online network.
            deterministic_actions = tf.argmax(network.q_out, axis=1)

            batch_size = tf.shape(network.obs_t_input)[0]

            random_actions = tf.random_uniform(shape=([batch_size]), 
                                              minval=0, 
                                              maxval=num_actions, 
                                              dtype=tf.int64)
            
            should_choose_random = (tf.random_uniform(tf.stack([batch_size]), 
                                                      minval=0, 
                                                      maxval=1, 
                                                      dtype=tf.float32) < self.epsilon)
            
            stochastic_actions = tf.where(should_choose_random, random_actions, deterministic_actions)

            self.output_actions = tf.cond(self.stochastic_t, 
                                     lambda: stochastic_actions, 
                                     lambda: deterministic_actions) 
            
            self.updates_eps_op = self.epsilon.assign(tf.cond(self.update_eps_t >= 0, 
                                                              lambda: self.update_eps_t, 
                                                              lambda: self.epsilon))
            
    def sample_act(self, sess, observations, stochastic=True, update_eps = -1):
        action, _ = sess.run([self.output_actions, self.updates_eps_op], 
                             feed_dict={self.network.obs_t_input: observations,
                                        self.stochastic_t: stochastic,
                                        self.update_eps_t: update_eps})
        return action

    def get_q(self, sess, observation):
        q = sess.run([self.network.q_out],
                     feed_dict={self.network.obs_t_input: observation})
        return q

    def train(self, sess, observation_t_with_g, primitive_action_t, intrinsic_reward_t, observation_tp1_with_g, done_t, importance_weights, online_q_tp1):
        td_error, _ = sess.run([self.network.td_error, self.network.optimize_op], 
                               feed_dict={self.network.obs_t_input: observation_t_with_g,
                                          self.network.action_t: primitive_action_t,
                                          self.network.reward_t: intrinsic_reward_t,
                                          self.network.obs_tp1_input: observation_tp1_with_g,
                                          self.network.importance_weights: importance_weights,
                                          self.network.done_t: done_t,
                                          self.network.online_q_tp1: online_q_tp1})
        return td_error



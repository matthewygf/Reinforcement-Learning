import tensorflow as tf
from gym import spaces
import utils.tf_util as U

active_fn = tf.nn.relu

class Q_network(object):
    def __init__(self, 
                 observation_space, 
                 num_outputs,
                 optimizer,
                 scope = '', # when we have multiple option, let's include this.
                 double = True, 
                 dueling = True,
                 gamma = 1.0,
                 hidden_nodes = [256],
                 data_format='channels_last',
                 grad_norm_clipping=None):
        with tf.variable_scope(scope+'/q_networks', reuse=tf.AUTO_REUSE):
            # set up placeholder 
            # observation should be scaled already
            if isinstance(observation_space, spaces.Dict):
                ob = observation_space.spaces['observation']
            else:
                ob = observation_space
            
            # the observation could be goal encoded. i.e. binary mask on the goal
            self.obs_t_input = tf.placeholder(tf.float32, shape=(None, )+ob.shape, name='obs_t')
            self.action_t = tf.placeholder(tf.int32, shape=[None], name='action')
            self.reward_t = tf.placeholder(tf.float32, shape=[None], name='reward')
            self.obs_tp1_input = tf.placeholder(tf.float32, shape=(None, )+ob.shape, name='obs_tp1')
            self.done_t = tf.placeholder(tf.float32, shape=[None], name='done')
            self.importance_weights = tf.placeholder(tf.float32, shape=[None], name='weight')

            # build online q network
            with tf.variable_scope('online_q', reuse=tf.AUTO_REUSE):
                c1 = tf.layers.conv2d(self.obs_t_input, 32, 8, 4, activation=active_fn, data_format=data_format)            
                c2 = tf.layers.conv2d(c1, 64, 4, 2, activation=active_fn, data_format=data_format)
                c3 = tf.layers.conv2d(c2, 64, 3, 1, activation=active_fn, data_format=data_format)
                flat = tf.layers.flatten(c3)

                # build action scores
                with tf.variable_scope('action_value'):
                    action_out = flat
                    for hidden_node in hidden_nodes:
                        action_out = tf.layers.dense(action_out, hidden_node, activation=None)
                        # open AI uses layer_normalization here
                        action_out = active_fn(action_out)
                    action_scores = tf.layers.dense(action_out, num_outputs, activation=None)
                
                # if dueling then we can compute the state values separately to get our final Q.
                if dueling:
                    with tf.variable_scope('state_value'):
                        state_out = flat
                        for hidden_node in hidden_nodes:
                            state_out = tf.layers.dense(state_out, hidden_node, activation=None)
                             # again open AI uses layer_normalization here
                            state_out = active_fn(state_out)
                        state_value_scores = tf.layers.dense(state_out, 1, activation=None)
                    action_scores_mean = tf.reduce_mean(action_scores, 1)
                    action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                    self.q_out = state_value_scores + action_scores_centered
                else:
                    self.q_out = action_scores
            
            # build target q network
            with tf.variable_scope('target_q', reuse=tf.AUTO_REUSE):
                c1_t = tf.layers.conv2d(self.obs_tp1_input, 32, 8, 4, activation=active_fn, data_format=data_format)            
                c2_t = tf.layers.conv2d(c1_t, 64, 4, 2, activation=active_fn, data_format=data_format)
                c3_t = tf.layers.conv2d(c2_t, 64, 3, 1, activation=active_fn, data_format=data_format)
                flat_t = tf.layers.flatten(c3_t)

                # build action scores
                with tf.variable_scope('action_value'):
                    action_out_t = flat_t
                    for hidden_node in hidden_nodes:
                        action_out_t = tf.layers.dense(action_out_t, hidden_node, activation=None)
                        # open AI uses layer_normalization here
                        action_out_t = active_fn(action_out_t)
                    action_scores_t = tf.layers.dense(action_out_t, num_outputs, activation=None)
                
                # if dueling then we can compute the state values separately to get our final Q.
                if dueling:
                    with tf.variable_scope('state_value'):
                        state_out_t = flat_t
                        for hidden_node in hidden_nodes:
                            state_out_t = tf.layers.dense(state_out_t, hidden_node, activation=None)
                             # again open AI uses layer_normalization here
                            state_out_t = active_fn(state_out_t)
                        state_value_scores_t = tf.layers.dense(state_out_t, 1, activation=None)
                    action_scores_mean_t = tf.reduce_mean(action_scores_t, 1)
                    action_scores_centered_t = action_scores_t - tf.expand_dims(action_scores_mean_t, 1)
                    self.q_out_t = state_value_scores_t + action_scores_centered_t
                else:
                    self.q_out_t = action_scores_t

            # receive online qnetwork estimate for the next state 
            self.online_q_tp1 = tf.placeholder(shape=self.q_out.shape, dtype=tf.float32)
            # calculate the q_value for our actual action
            self.q_t_selected = tf.reduce_sum(self.q_out * tf.one_hot(self.action_t, num_outputs), 1)

            if double:
                q_tp1_with_online = tf.argmax(self.online_q_tp1, 1)
                q_tp1_best = tf.reduce_sum(self.q_out_t * tf.one_hot(q_tp1_with_online, num_outputs), 1)
            else:
                q_tp1_best = tf.reduce_max(self.q_out_t, 1)

            q_tp1_best_done = (1.0 - self.done_t) * q_tp1_best

            # immediate reward * discount estimate
            q_t_selected_target = self.reward_t + gamma * q_tp1_best_done

            # compute td error
            # actual q value - return q value from environment and discounted
            self.td_error = self.q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors = U.huber_loss(self.td_error)
            weighted_error = tf.reduce_mean(self.importance_weights * errors)

            # we are just under our default scope name
            online_q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + '/online_q')
            gradients = optimizer.compute_gradients(weighted_error, var_list=online_q_vars)

            # get our optimize_op
            if grad_norm_clipping is not None:
                for i, (grad, var) in enumerate(gradients):
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
                self.optimize_op = optimizer.apply_gradients(gradients)
            else:
                self.optimize_op = optimizer.apply_gradients(gradients)
            
            # get our update op for target q
            update_target_ops = []
            target_q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + '/target_q')
            for var, var_target in zip(sorted(online_q_vars, key=lambda v: v.name),
                                       sorted(target_q_vars, key=lambda v: v.name)):
                update_target_ops.append(var_target.assign(var))
            # group the updates op to one op
            self.update_target_op = tf.group(*update_target_ops)

            



                
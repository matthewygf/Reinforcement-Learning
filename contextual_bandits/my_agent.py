import tensorflow as tf

class my_agent():
    def __init__(self, num_states, num_actions):
        # feeding in one input
        self.num_states = num_states
        self.num_actions = num_actions
        self.current_s = tf.placeholder(dtype=tf.int32, shape=[1])
        states_one_hot = tf.one_hot(self.current_s, num_states)
        # so i have tried two hidden layers, it seems to be exploiting a lot, why ?
        # overfit to something occasionally winning without exploring too much ? 
        output = tf.layers.dense(states_one_hot, num_actions, activation=tf.nn.sigmoid)
        self.output = tf.reshape(output, shape=[-1])
        self.action_op = tf.arg_max(self.output, 0)

        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        # that one weight that is responsible for choosing the action
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.train_op = optimizer.minimize(self.loss)

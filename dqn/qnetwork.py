import tensorflow as tf

class Qnetwork:
    """
        Dueling Q network takes observation (visual) to predict how what action we should take.
        Key: q values = value function + advantage function

        NOTE: author from DeepMind Dueling Q network doesn't split the final network.
        here from the blog i follow, see main.py
        the op split the network as it seems to work well for the problem.
    """
    def __init__(self, input_image_size, hidden_size, num_actions):
        """
            params:
                input_image_size - [width, height, channel] because i have no GPU lol
                hidden_size - fc layer units
        """
        self.image_inputs = tf.placeholder(tf.float32, shape=[None, input_image_size[0], input_image_size[1], input_image_size[2]], name='inputs')

        # conv layers 2d
        self.conv1 = tf.layers.conv2d(self.image_inputs, 32, [10, 5], strides=[4, 5])
        self.conv2 = tf.layers.conv2d(self.conv1, 64, [6, 4], strides=[5, 3])
        self.conv3 = tf.layers.conv2d(self.conv2, 64, [3, 5], strides=[1, 1])
        self.conv4 = tf.layers.conv2d(self.conv3, hidden_size, [8, 6], strides=[1, 1])
        
        # NOTE: this is the modified part from the dueling DQN
        # split into two streams, advantage and value
        self.advantages, self.values = tf.split(axis=3, value=self.conv4, num_or_size_splits=2)
        self.advantages_oned = tf.layers.flatten(self.advantages)
        self.values_oned = tf.layers.flatten(self.values)

        # advantage function say how much better taking a certain action would be compared to others
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.advantage_weights = tf.Variable(xavier_init([hidden_size // 2, num_actions]))
        # value function say how good it is to be in any given state
        self.value_weights = tf.Variable(xavier_init([hidden_size // 2, 1]))

        self.advantage_output = tf.matmul(self.advantages_oned, self.advantage_weights)
        self.value_output = tf.matmul(self.values_oned, self.value_weights)

        # get our final Q
        # https://arxiv.org/pdf/1511.06581.pdf equation - 9
        self.q_output = self.value_output + tf.subtract(self.advantage_output, tf.reduce_mean(self.advantage_output, axis=1, keep_dims=True))
        
        self.predict = tf.argmax(self.q_output, axis=1)
        
        # sum square diff between target & prediction
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, on_value=1.0, off_value=0.0)
        self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot), axis=1)
        # calculate the td error
        # https://arxiv.org/pdf/1511.06581.pdf equation - 4
        self.td_error = tf.square(self.target_q - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        # train op
        self.train_op = self.optimizer.minimize(self.loss)


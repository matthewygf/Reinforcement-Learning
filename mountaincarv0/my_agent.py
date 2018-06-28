import tensorflow as tf

class my_agent:
    """
        agent will take the states as input, calculate our feed forward layer

        it will also take the reward to calculate the loss function 
    """
    def __init__(self, num_states, num_actions, lr=0.002):
        # Take a batch of states
        # construct our feed forward network
        self.current_s = tf.placeholder(tf.float32, shape=[None, num_states])
        h_layer = tf.layers.dense(self.current_s, 10, activation=tf.nn.sigmoid, name="simple/hidden")
        self.output = tf.layers.dense(h_layer, num_actions, activation=tf.nn.sigmoid, name="simple/output")
        # choose the action
        self.action_op = tf.argmax(self.output, 1)

        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        
        # pull that responsibile weights from output
        # i am not too sure why we need to do this...
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        # calculate our loss
        self.loss = - tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+"_placeholder")
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
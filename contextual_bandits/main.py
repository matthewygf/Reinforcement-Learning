import numpy as np
import tensorflow as tf
from contextual_bandit import contextual_bandit
from my_agent import my_agent

"""
    adopted from
        https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c
    contextual bandit is not a full RL problem,
    because action does not affect the state. whereas full RL, action affect the state we are in.
    
    state -> action -> reward
    |                      |
    |______________________|
"""

EPS = 0.3
EPISODES = 20000

def ep_greedy(sess, state, agent):
    """
        policy ep-greedy choose an action randomly or
        choose an action from our neural network by feeding in current state

        params:
            sess - tensorflow session
            state - in this case state is which bandit we are in.
            agent - agent will choose action based on state
    """
    if np.random.uniform(0, 1) < EPS:
        action = np.random.randint(agent.num_actions)
    else:
        # run the agent action_op
        action = sess.run(agent.action_op, feed_dict={agent.current_s:[state]})
    return action


def main():
    tf.reset_default_graph()

    # init the bandit
    Bandits = contextual_bandit()
    c_agent = my_agent(Bandits.num_bandits, Bandits.num_actions)

    # keep track of our rewards
    total_rewards = np.zeros((Bandits.num_bandits, Bandits.num_actions))

    # init tf variables()
    init_op = tf.global_variables_initializer()
    # peek into our weights 
    trainables = tf.trainable_variables()
    print(trainables[-2])
    weights = trainables[-2]

    with tf.Session() as sess:
        sess.run(init_op)
        t = 0

        while t < EPISODES:
            # choose an action based on policy 
            state = Bandits.start()
            action = ep_greedy(sess, state, c_agent)
            reward = Bandits.step(action)

            # init the dictionary to feed to our network
            feed_dict = {c_agent.reward_holder: [reward], c_agent.action_holder: [action], c_agent.current_s: [state]}
            
            # train our network and peek into our weights
            _, r_weight = sess.run([c_agent.train_op, weights], feed_dict=feed_dict)

            # keep track of our score
            total_rewards[state, action] += reward

            if t % 500 == 0:
                print("Mean reward for the %s bandits: %s" % (str(Bandits.num_bandits), str(np.mean(total_rewards, axis=1))))

            t += 1

    for b in range(Bandits.num_bandits):
        print ("the agent thinks action %d for bandit %d is the most promising..." % (np.argmax(r_weight[b])+1, b+1))
        if np.argmax(r_weight[b]) == np.argmin(Bandits.bandits[b]):
            print("and it was right")
        else:
            print("and it was wrong lol")

if __name__ == '__main__':
    main()



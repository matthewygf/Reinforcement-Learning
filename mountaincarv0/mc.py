import gym
import gym.spaces
import numpy as np
import tensorflow as tf

from my_agent import my_agent

"""
    MountainCar-v0 
    Monte-carlo control approach to solve this env.
    Learn by at the end of the episodes
    with policy gradient.

    adopted from 
    https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
"""

GAMMA = 0.99

def discount_reward(rewards):
    num_timestep = len(rewards)
    discounted_r = np.zeros_like(rewards)
    # start from reverse to 0
    discount = 0
    for t in range(num_timestep-1, -1, -1):
        # from this state onwards, immediate reward plus future discounted
        discount = rewards[t] + (discount * np.power(GAMMA, t))
        discounted_r[t] = discount
    return discounted_r

def main():
    tf.reset_default_graph()

    # make our environment
    env = gym.make('MountainCar-v0')
        
    # get our available actions 0 - n-1
    action_space = env.action_space
    states_space = env.observation_space

    # run 5000 episodes
    # update our agent every 5 episodes
    episodes = 5000
    update_frequency = 5

    # init our agent
    m_agent = my_agent(states_space.shape[0], action_space.n)
    global_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(global_op)
        total_reward = []
        total_update = 0
        gradients_buffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(gradients_buffer):
            gradients_buffer[ix] = grad * 0

        for i_episode in range(episodes):
            # start observation (state)
            observation = env.reset()
            ep_reward = 0
            ep_history = []
            # take 999 steps maximum
            for t in range(999):
                env.render()

                # we should use neural network to choose for us.
                action_dist = sess.run(m_agent.output, feed_dict={m_agent.current_s:[observation]})
                # randomly produce an action in uniform
                action = np.random.choice(action_dist[0])
                action = np.argmax(action_dist == action)

                # lets take a step
                next_observation, reward, done, info = env.step(action) 
                # s, a, r, s'
                ep_history.append([observation, action, reward, next_observation])
                observation = next_observation
                ep_reward += reward

                # wait till we are done.
                # then we update our network at the end.
                # so we get the true policy value 
                if done:
                    # calculate the total discounted reward
                    # http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MC-TD.pdf
                    # third column is our reward column
                    ep_history_array = np.array(ep_history)
                    ep_history_array[:, 2] = discount_reward(ep_history_array[:, 2])                    
                    
                    # train our network
                    # 1. calculate gradients 
                    feed_dict = {m_agent.reward_holder: ep_history_array[:,2],
                                m_agent.action_holder: ep_history_array[:,1],
                                m_agent.current_s: np.vstack(ep_history_array[:,0])}
                    gradients, indexes = sess.run([m_agent.gradients, m_agent.indexes], feed_dict=feed_dict)

                    for idx, grad in enumerate(gradients):
                        gradients_buffer[idx] += grad

                    # 2. apply to our variables
                    if i_episode % update_frequency == 0 and i_episode != 0:
                        total_update += 1
                        feed_dict = dict(zip(m_agent.gradient_holders, gradients_buffer))
                        print("updating neural network weights ... %d" % total_update)
                        _ = sess.run(m_agent.train_op, feed_dict=feed_dict)
                        # reset our gradients buffer
                        for ix, grad in enumerate(gradients_buffer):
                            gradients_buffer[ix] = grad * 0

                    total_reward.append(ep_reward)
                    break
            if i_episode % 100 == 0:
                # calculate the last 100 mean reward
                print(np.mean(total_reward[-100:]))

if __name__ == '__main__':
    main()


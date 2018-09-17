import gym
import tensorflow as tf
from deepq.models import Q_network
from deepq.controller import MetaController
from deepq.replay_buffer import ReplayBuffer
from utils.schedules import LinearSchedule
import envs.env_wrapper as wrapper
import utils.tf_util as U
from datetime import datetime
import numpy as np
import utils.logger as L
def main():
  L.configure('/home/metalabadmin/exp/freeway', format_strs=['stdout', 'csv', 'tensorboard'])
  env = gym.make('Freeway-v0')
  env = wrapper.wrap_deepmind(env, frame_stack=True, scale=True)

  optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
  network = Q_network(env.observation_space, env.action_space.n, optimizer, gamma=0.99, scope='freeway')
  m_controller = MetaController(network, env.action_space.n)
  # Create the schedule for exploration starting from 1.
  exploration = LinearSchedule(schedule_timesteps=int(0.1 * 1e7),
                                initial_p=1.0,
                                final_p=0.02)
  replay = ReplayBuffer(50000)
  # get default tf_session
  sess = U.get_session()
  U.initialize()
  sess.run(m_controller.network.update_target_op)
  step = 0
  episodes = 0
  rewards = 0
  mean_100ep_reward = 0
  total_reward = []
  saved_mean_reward = None
  ob = env.reset()

  while step <= 1e7:
    ep = exploration.value(step)
    ob_reshaped = np.reshape(ob, (1, )+env.observation_space.shape)
    act = m_controller.sample_act(sess, ob_reshaped, update_eps=ep)[0]
    ob_tp1, reward_t, done_t, info = env.step(act)
    env.render()
    rewards += reward_t
    replay.add(ob, act, reward_t, ob_tp1, float(done_t))
    ob = ob_tp1

    # train every 4 steps
    if step >= 1000 and step % 4 == 0:
      obs, acts, rewards_t, obs_tp1, dones_t = replay.sample(64)
      weights, batch_idxes = np.ones_like(rewards_t), None
      # get q estimate for tp1 as 'supervised'
      obs_tp1_reshaped = np.reshape(obs_tp1, (64, )+env.observation_space.shape)
      q_tp1 = m_controller.get_q(sess, obs_tp1_reshaped)[0]
      td_error = m_controller.train(sess, obs, acts, rewards_t, obs_tp1, dones_t, weights, q_tp1)

    step += 1 

    if step >= 1000 and step % 1000 == 0:
      sess.run(m_controller.network.update_target_op)
    
    if done_t:
      ob = env.reset()
      total_reward.append(rewards)
      episodes += 1
      rewards = 0
      print('step %d done %s, ep %.2f' % (step, str(done_t), ep))
      mean_100ep_reward = round(np.mean(total_reward[-101:-1]), 1)
      if episodes % 10 == 0 and episodes != 0:
        print('date time %s' % str(datetime.now()))
        L.record_tabular("steps", step)
        L.record_tabular("episodes", episodes)
        L.record_tabular("mean 100 episode reward", mean_100ep_reward)
        L.dump_tabular()
    
    if step % 1000 == 0:
      if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
        L.log("Saving model due to mean reward increase: {} -> {}".format(
                            saved_mean_reward, mean_100ep_reward))
        U.save_variables('./freewaymodel.ckpt')
        model_saved = True
        saved_mean_reward = mean_100ep_reward
  

if __name__ == '__main__':
  main()
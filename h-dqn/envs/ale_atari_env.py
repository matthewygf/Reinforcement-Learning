# taken from gym atari_env.py
# needed to modify as we want to set goals in the environment
# specifically only for image at the moment
# ===================================================================================================================

import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import os

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install Atari dependencies by running 'pip install gym[atari]'.)".format(e))

# HACK At the moment this env is assumed to be montezuma_revenge
# might as well be Montezuma_revenge env :/
class AtariEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, monitor, frameskip=(2, 5), repeat_action_probability=0.):
        self.game_path = atari_py.get_game_path(monitor.game_name)

        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist'%(monitor.game_name, self.game_path))
        
        self._obs_type = 'image' # HACK to image for now.
        self.frameskip = frameskip
        self.ale = atari_py.ALEInterface()
        self.viewer = None
        # added monitor to keep track of things
        self.monitor = monitor

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        self.seed_and_load_rom()

        self._action_set = self.ale.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        # goals specific
        self._goals_set = monitor.goals_set_small # 84x84
        self._goals_center = monitor.goals_center
        self.goals_space = spaces.Discrete(len(self._goals_set))
        self.desired_goal = -1 # we set and tell the agent to achieve this desired_goal.
        self.achieved_goal = -1 # we should keep track of which goal it currently achieved. 
        self.goals_history = [] # can keep track of how it achieved the set of goals to the currently achieved_goal

        # we need to calculate whether agent achieve the goal so we need to keep track of agent loc
        # HACK only montezuma_revenge specific right now
        if monitor.game_name == 'montezuma_revenge':
            self.agent_origin = [42, 33]
            self.agent_last_x = 42
            self.agent_last_y = 33

        (screen_width, screen_hight) = self.ale.getScreenDims()

        self.init_screen = self.ale.getScreenGrayscale()

        # Don't think i will use this 
        if self._obs_type == 'ram':
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(low=0, high=255, shape=(screen_hight, screen_width, 3), dtype=np.uint8),
                'achieved_goal': spaces.Discrete(1),
                'desired_goal': spaces.Discrete(1)
            })
        elif self._obs_type == 'image':
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(low=0, high=255, shape=(screen_hight, screen_width, 3), dtype=np.uint8),
                'achieved_goal': spaces.Discrete(1),
                'desired_goal': spaces.Discrete(1)
            })
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def seed_and_load_rom(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)
        self.ale.loadROM(self.game_path)
        return [seed1, seed2]
 
    def step(self, a):
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.ale.act(action)

        ob = self._get_obs()
        # TODO: here returns the extrinsic reward
        # we should look at ways for intrinsic reward
        return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def _get_obs(self):
        # TODO : how to calculate achieved / desired
        if self._obs_type == 'ram':
            raise NotImplementedError
        elif self._obs_type == 'image':
            img = self._get_image()
        ob = {}
        ob['observation'] = img
        ob['achieved_goal'] = self.achieved_goal
        ob['desired_goal'] = self.desired_goal
        return ob

    @property
    def _n_actions(self):
        return len(self._action_set)

    """
    def _get_ram(self):
        return to_ram(self.ale)
    """

    # return: (states, observations)
    def reset(self):
        self.ale.reset_game()
        self.desired_goal = -1
        self.achieved_goal = -1
        self.goals_history = []
        # HACK for montezuma
        if self.monitor.game_name == 'montezuma_revenge':
            self.agent_last_x = self.agent_origin[0]
            self.agent_last_y = self.agent_origin[1]
        return self._get_obs()

    def _get_image(self, show_goals=True):
        screen = self.ale.getScreenRGB2()
        if show_goals and self.monitor is not None:
            for goal in self.monitor.goals_set_large:
                x1 = goal[0][0]
                x2 = goal[1][0]
                y1 = goal[0][1]
                y2 = goal[1][1]
                screen[y1,x1:x2,:] = 255
                screen[y2,x1:x2,:] = 255
                screen[y1:y2,x1,:] = 255
                screen[y1:y2,x2,:] = 255
        return screen

    def render(self, mode='human'):
        img = self._get_image() # TODO: Fix this.
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)

    # Goals - intrinsic reward ? 
    def compute_reward(self, achieved_goal, desired_goal, info):
        assert achieved_goal < len(self._goals_center)
        assert desired_goal < len(self._goals_center)

        if (achieved_goal == -1):
            achieved_goal_location = self.agent_origin
        else:
            # get our last achieved goal last location
            achieved_goal_location = self._goals_center[achieved_goal]
        desired_goal_location = self._goals_center[desired_goal]
        agent_x, agent_y = self.get_agent_loc()

        desired_goal_dist_diff_x = desired_goal_location[0] - agent_x
        desired_goal_dist_diff_y = desired_goal_location[1] - agent_y
        desired_goal_dist = np.sqrt(np.power(desired_goal_dist_diff_x, 2) + np.power(desired_goal_dist_diff_y, 2))

        achieved_goal_dist_diff_x = achieved_goal_location[0] - agent_x
        achieved_goal_dist_diff_y = achieved_goal_location[1] - agent_y
        achieved_goal_dist = np.sqrt(np.power(achieved_goal_dist_diff_x, 2) + np.power(achieved_goal_dist_diff_y, 2))

        between_goal_dist_diff_x = desired_goal_dist[0] - achieved_goal_dist[0]
        between_goal_dist_diff_y = desired_goal_dist[1] - achieved_goal_dist[1]
        between_goal_dist = np.sqrt(np.power(between_goal_dist_diff_x, 2) + np.power(between_goal_dist_diff_y, 2))

        # normalize how far the agent is to the next goal.
        return 0.001 * (achieved_goal_dist - desired_goal_dist) / between_goal_dist

    def get_agent_loc(self):
        # HACK: Montezuma_revenge specific
        img = self.ale.getScreenRGB2()
        agent = [200, 72, 72] # RGB values of the agent
        mask = np.zeros(np.shape(img))
        mask[:,:, 0] = agent[0] # R channel
        mask[:,:, 1] = agent[1] # G channel
        mask[:,:, 2] = agent[2] # B channel

        # findout where the agent is
        diff = img - mask 
        indexs = np.where(diff == 0)
        # makes everything black except agent to white
        diff[np.where(diff < 0)] = 0
        diff[np.where(diff > 0)] = 0
        diff[indexs] = 255

        if (np.shape(indexs[0])[0] == 0):
            print("---------------------- diff index shape 0 is 0")
            mean_x = self.agent_last_x
            mean_y = self.agent_last_y
        else:
            mean_y = np.sum(indexs[0]) / np.shape(indexs[0])[0]
            mean_x = np.sum(indexs[1]) / np.shape(indexs[1])[0]
        self.agent_last_x = mean_x
        self.agent_last_y = mean_y
        return (mean_x, mean_y)

    def reached_goal(self, desired_goal):
        desired_goal_pos = self._goals_set[desired_goal]
        screen_with_goals = self.init_screen
        current_state_screen = self.ale.getScreenGrayscale()
        count = 0
        for x in range(desired_goal_pos[0][0], desired_goal_pos[1][0]):
            for y in range(desired_goal_pos[0][1], desired_goal_pos[1][1]):
                if screen_with_goals[y][x] != current_state_screen[y][x]: # screen is [y,x,c]
                    count += 1
        # 30 is total number of pixels of the agent
        if float(count) / 30 > 0.3:
            # consider agent has overlap enough pixels
            self.goals_history.append(desired_goal)
            self.achieved_goal = desired_goal
            return True
        return False

ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}
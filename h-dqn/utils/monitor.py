import tensorflow as tf

tf.app.flags.DEFINE_string('log_dir', '/tmp/log_dir',
                           """Directory where to write event logs """)

tf.app.flags.DEFINE_string('ckpt_dir', 'tmp/ckpt_dir',
                            """Directory where to write model ckpt""")

tf.app.flags.DEFINE_string('game_name', 'montezuma_revenge',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('num_timesteps', int(2e6),
                            """total num of timesteps""")

tf.app.flags.DEFINE_integer('ckpt_freq', int(1e4),
                            """when to ckpt our model""")

tf.app.flags.DEFINE_bool('load_model', False,
                            """whether to load model from ckpt_dir""")

tf.app.flags.DEFINE_integer('print_freq', int(20),
                            """when to ckpt our model""")

# GOALS DEFINED
# in 210, 160 pixels 
LOWER_RIGHT_LADDER = [(131, 172), (140, 178)]
KEY = [(12, 100), (24, 120)]
RIGHT_DOOR = [(130, 50), (144, 80)]
# in 84, 84 pixels ?
# from https://github.com/hoangminhle/hierarchical_IL_RL.git
LOWER_RIGHT_LADDER_SMALL = [(69, 68), (73, 71)]
KEY_SMALL = [(7, 41), (11, 45)]
RIGHT_DOOR_SMALL = [(70, 20), (73, 35)]

class Monitor(object):
    def __init__(self, params):
        self.params = params
        self.log_dir = self.params.log_dir
        self.ckpt_dir = self.params.ckpt_dir
        self.game_name = self.params.game_name
        self.num_timesteps = self.params.num_timesteps
        self.ckpt_freq = self.params.ckpt_freq
        self.print_freq = self.params.print_freq
        self.load_model = self.params.load_model

        self.goals_set_large = [LOWER_RIGHT_LADDER, KEY, LOWER_RIGHT_LADDER, RIGHT_DOOR]
        self.goals_set_small = [LOWER_RIGHT_LADDER_SMALL, KEY_SMALL, LOWER_RIGHT_LADDER_SMALL, RIGHT_DOOR_SMALL]

        self.calculate_goals_center()

    def calculate_goals_center(self, use_small=True):
        """
            Args
            use_small - use the 84x84 pixels values of the goals
        """
        goals_centers = []

        if use_small:
            goals = self.goals_set_small
        else:
            goals = self.goals_set_large

        for goal in goals:
            x_mid = float(goal[0][0] + goal[1][0]) / 2
            y_mid = float(goal[0][1] + goal[1][1]) / 2
            goals_centers.append((x_mid, y_mid))
        
        self.goals_center = goals_centers

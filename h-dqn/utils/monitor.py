import tensorflow as tf

tf.app.flags.DEFINE_string('log_dir', '/tmp/log_dir',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('game_name', 'montezuma_revenge',
                           """Directory where to write event logs """
                           """and checkpoint.""")

class Monitor(object):
    def __init__(self, params):
        self.params = params
        self.log_dir = self.params.log_dir
        self.game_name = self.params.game_name
import tensorflow as tf

# learn_rate:
# warmup_steps
# warmup_mode
# decay_mode
# init_rate
# start_decay_step
# decay_steps
# decay_factor
class LearningRateBuilder:
    def __init__(self, config, global_step, name="learning_rate"):
        self.config = config
        self.name = name
        self.global_step = global_step




    def __call__(self):

        with tf.variable_scope(self.name):
            rate = tf.get_variable(name="rate", dtype=tf.float32, initializer=self.config.init_rate)
            print(rate.get_shape())
            # rate = self._warmup(rate)
            #rate = self._decay()rate
            rate = self._decay(self.config.init_rate)
        return rate


    def _warmup(self, start_rate):

        if self.config.warmup_mode == "None":
            return start_rate
        elif self.config.warmup_mode == "t2t":
            warmup_steps = tf.to_float(self.config.warmup_steps)
            warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
            #warmup_factor ** (warmup_steps - global_step)
            inv_decay = tf.pow(warmup_factor, tf.constant(warmup_steps) - self.global_step)
        else:
            raise NotImplementedError

        rate = tf.cond(self.global_step < warmup_steps,
                       lambda: inv_decay * start_rate,
                       lambda: start_rate,
                       name="learning_rate_warm_up_cond")
        return rate

    def _decay(self, start_rate):

        if self.config.decay_mode == "None":
            return start_rate
        elif self.config.decay_mode == "exp":
            rate = tf.cond(self.global_step < self.config.start_decay_step,
                           lambda: start_rate,
                           lambda: tf.train.exponential_decay(start_rate,
                                                              self.global_step - self.config.start_decay_step,
                                                              self.config.decay_steps,
                                                              self.config.decay_factor,
                                                              name="learning_rate_delay_cond"))
        #elif self.config.decay_mode == "inverse":
        #    rate = tf.cond(self.global_step < self.config.start_decay_step,
        #                   lambda: start_rate,
        #                   lambda: tf.train.inverse_time_decay(start_rate,
        #                                                      self.global_step - self.config.start_decay_step,
        #                                                      self.config.decay_steps,
        #                                                      self.config.decay_factor))
        elif self.config.decay_mode == "cosine":
            print("learning rate decay")
            rate = tf.train.linear_cosine_decay(start_rate,
                                                self.global_step,
                                                self.config.start_decay_step,
                                                self.config.t_mul,
                                                self.config.m_mul,
                                                self.config.alpha,
                                                name="learning_rate_delay_cond_cosine")
        return rate
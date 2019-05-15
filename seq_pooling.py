import tensorflow as tf

class SeqPooling:

    def __init__(self, config, name, fetch_info=None):
        self.config = config
        self.name = name
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.use_len = False
        self.fetch_info = fetch_info

    def __call__(self, input, input_len):
        self.batch_size = tf.shape(input)[0]
        self.hidden_state_dim = input.get_shape().as_list()[-1]

        if self.config.type == "max_mean":
            output = self._max_mean(input, input_len)
        elif self.config.type == "weighted_mean":
            output = self._weighted_mean(input, input_len)
        else:
            raise NotImplementedError("SeqPooling")
        return output

    def _weighted_mean(self, input, input_len):

        # print(self.hidden_state_dim)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            W_u_q = tf.get_variable(shape=[self.config.attn_size, self.hidden_state_dim],
                                    initializer=self.initializer, name="W_u_q")
            W_v_q = tf.get_variable(shape=[self.config.attn_size, self.config.q_emb_size], \
                                    initializer=self.initializer, name="W_v_q") #[attn_size, q_emb_size]
            V_r_q = tf.get_variable(shape=[self.config.q_emb_size],
                                    initializer=self.initializer, name="V_r_q") #[]
            v_q = tf.get_variable(shape=[self.config.attn_size],
                                  initializer=self.initializer, name="v_q")

            # print(query_encoding.get_shape())
            W_u_q_t = tf.tile(tf.expand_dims(tf.transpose(W_u_q), axis=0),
                              multiples=[self.batch_size, 1, 1]) #[batch size, hidden_state_dim, attn_size]
            # print(W_u_q_t.get_shape())
            #query_encoding: [batch size, q max length, hidden_state_dim]
            t_1 = tf.matmul(input, W_u_q_t) #[batch, max length, attn_size]
            # print(t_1.get_shape())
            # v_q_t = tf.tile(tf.expand_dims(tf.expand_dims(v_q, axis=1), 0),
            #                 multiples=[self.batch_size, 1, 1]) #[attn_size, 1]
            V_r_q_t = tf.expand_dims(V_r_q, axis=0) #[1, q_emb_size]
            t_2 = tf.matmul(V_r_q_t, W_v_q, transpose_b=True) #[1, attn_size]
            t_2 = tf.expand_dims(t_2, 0) #[1, 1, attn_size]
            v_q_t = tf.tile(tf.expand_dims(tf.expand_dims(v_q, axis=1),axis=0),
                            multiples=[self.batch_size, 1, 1])#[batch, attn size, 1]
            s = tf.squeeze(tf.matmul(tf.tanh( t_1 + t_2), v_q_t), axis=[2]) #[batch_size, q max lengh]

            #fill with negative infinite
            if self.use_len:
                s = tf.where(tf.sequence_mask(input_len), s, tf.fill(tf.shape(s), -float('inf')))

            a = tf.nn.softmax(s, axis=-1) #[batch_size, q max length]

            if self.fetch_info is not None:
                self.fetch_info.add_info("%s_q_pooling_softmax" %self.name, a)

            r = tf.reduce_sum(input * tf.expand_dims(a, -1), axis=1) #[batch_size, dim]
            # r = tf.reshape(tf.squeeze(r, axis=[]), shape=[self.batch_size, self.hidden_state_dim])
            # print("test ", r.get_shape())
            # exit(1)
            # r = tf.reshape(r, shape=[self.batch_size, self.hidden_state_dim])

        return r

    def _max_mean(self, input, input_len):

        input_max = tf.reduce_max(input, axis=-2)

        if self.use_len:

            raise NotImplementedError("max mean")
        else:
            input_mean = tf.reduce_mean(input, axis=-2)

        if self.fetch_info is not None:
            self.fetch_info("max_mean_pooling_argmax",tf.argmax(input, axis=-2))

        # words_indices = tf.cast(tf.sequence_mask(sents_len), tf.float32)
        # words_indices = tf.tile(tf.expand_dims(words_indices, -1),
        #                         multiples=[1,1,1,sents_encoding.get_shape()[-1]])
        # sent_mean = tf.reduce_mean(sents_encoding, axis=2)
        output = tf.concat([input_max, input_mean], axis=-1)
        return output


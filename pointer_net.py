import tensorflow as tf
from rnn_encoding import SeqPooling

'''
dim: hidden state size
q_emb_size: size of v
attn_size: size of matrix used to calculate attention

'''
class MRCPointerNet:


    def __init__(self, config, name, fetch_info=None):
        self.config = config
        self.name = name
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.use_len = False
        self.fetch_info = fetch_info


    def get_basic_cell(self, name):
        basic_cell = None
        if self.config.cell == "lstm":
            basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_state_dim, name=name)
        elif self.config.cell == "gru":
            basic_cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_state_dim, name=name)
        else:
            raise NotImplementedError

        return basic_cell

    def __call__(self, query_encoding, chunk_encoding, query_len, chunk_len):

        self.hidden_state_dim = query_encoding.get_shape().as_list()[-1]
        self.batch_size = tf.shape(query_encoding)[0]

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # outputs = self._query_pooling(query_encoding, query_len)
            basic_rnn_cell = self.get_basic_cell(name=self.config.cell)

            print("lstm state size ", basic_rnn_cell.state_size)
            #
            query_vector = self._query_pooling(query_encoding, query_len)
            init_state = self.make_initial_state(query_vector, basic_rnn_cell)
            a1, p1, score1, state1, output1 = self._one_step_decoding(init_state, chunk_encoding, basic_rnn_cell, chunk_len)

            if self.fetch_info is not None:
                self.fetch_info.add_info("%s_decoding_1_softmax" %self.name, a1)

            a2, p2, score2, state2, output2 = self._one_step_decoding(state1, chunk_encoding, basic_rnn_cell, chunk_len)

            if self.fetch_info is not None:
                self.fetch_info.add_info("%s_decoding_2_softmax" % self.name, a2)

            return a1, a2, p1, p2, query_vector, score1, score2




    def make_initial_state(self, query_vector, rnn_cell):

        if self.config.cell == "lstm":
            init_state = tf.contrib.rnn.LSTMStateTuple(query_vector, tf.zeros(shape=[self.batch_size, self.hidden_state_dim]))
            return init_state
        elif self.config.cell == "gru":
            init_state = query_vector
            return init_state
        else:
            raise NotImplementedError

    def _query_pooling(self, query_encoding, query_len):

        # print(self.hidden_state_dim)
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
        t_1 = tf.matmul(query_encoding, W_u_q_t) #[batch, max length, attn_size]
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
            s = tf.where(tf.sequence_mask(query_len), s, tf.fill(tf.shape(s), -float('inf')))

        a = tf.nn.softmax(s, axis=-1) #[batch_size, q max length]

        if self.fetch_info is not None:
            self.fetch_info.add_info("%s_q_pooling_softmax" %self.name, a)

        r = tf.reduce_sum(query_encoding * tf.expand_dims(a, -1), axis=1) #[batch_size, dim]
        # r = tf.reshape(tf.squeeze(r, axis=[]), shape=[self.batch_size, self.hidden_state_dim])
        # print("test ", r.get_shape())
        # exit(1)
        # r = tf.reshape(r, shape=[self.batch_size, self.hidden_state_dim])
        return r

    def _one_step_decoding(self, previous_state, chunk_encoding, rnn_cell, chunk_len):

        #previous_state [batch_size, hidden_state_dim]
        if self.config.cell == "lstm":
            state_to_compute_attention = previous_state.c
        elif self.config.cell == "gru":
            state_to_compute_attention = previous_state
        else:
            raise NotImplementedError


        chunk_dim = chunk_encoding.get_shape().as_list()[-1]
        batch_size = tf.shape(chunk_encoding)[0]

        W_h_p = tf.get_variable(shape=[self.config.attn_size, chunk_dim],
                                initializer=self.initializer, name="W_h_p") #[attn_size, chunk_dim]
        W_h_a = tf.get_variable(shape=[self.config.attn_size, self.hidden_state_dim],
                                initializer=self.initializer, name="W_h_a") #[attn, hidden dim]
        v_p = tf.get_variable(shape=[self.config.attn_size],
                              initializer=self.initializer, name="v_p") #[attn_size]

        W_h_p_t = tf.tile(tf.expand_dims(tf.transpose(W_h_p), axis=0), multiples=[batch_size, 1, 1])
        t_1 = tf.matmul(chunk_encoding, W_h_p_t) #[batch, max length, attn_size]
        t_2 = tf.matmul(state_to_compute_attention, W_h_a, transpose_b=True) #[batch, attn_size]
        t_2 = tf.expand_dims(t_2, axis=1) #[batch, 1, attn_size]
        t_sum = tf.tanh(t_1 + t_2) #[batch, length, attn_size]
        v_p_t = tf.tile(tf.expand_dims(tf.expand_dims(v_p, axis=1), axis=0),
                        multiples=[self.batch_size, 1, 1]) #[batch size, attn_size, 1]
        s = tf.squeeze(tf.matmul(t_sum, v_p_t), axis=[2]) #[batch, length]

        if self.use_len:
            s = tf.where(tf.sequence_mask(chunk_len), s, tf.fill(tf.shape(s), -float('inf')))

        a = tf.nn.softmax(s) #[batch, length]

        p = tf.argmax(a, -1)
        c = tf.reduce_sum(chunk_encoding * tf.expand_dims(a, -1), axis=1) #[batch, dim]
        # c = tf.reshape(c, shape=[self.batch_size, chunk_dim])
        print("c ", c.get_shape())
        output, new_state = rnn_cell(c, previous_state)

        return a, p, s, new_state, output




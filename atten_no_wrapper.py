#implement attention without using wrapper

import tensorflow as tf
from attn_base import AttentionBase
from var_dropout import variational_rnn_dropout, independent_dropout
from rnn_encoding import RNNEncoder


class AttentionRNetNoWrapper(AttentionBase):

    def __init__(self, config, name, fetch_info=None):
        self.config = config
        self.name = name
        self.initializer = tf.random_normal_initializer(stddev=0.5)
        self.use_len = False
        self.fetch_info = fetch_info




    def __call__(self, memory, inputs, memory_len, inputs_len):
        #this implementation is a simplification of rnet

        memory_dim = memory.get_shape()[-1]
        inputs_dim = inputs.get_shape()[-1]
        batch_size = tf.shape(inputs)[0]
        i_len = tf.shape(inputs)[1]
        m_len = tf.shape(memory)[1]
        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            #self.config.depth
            W_u_Q = tf.get_variable(shape=[self.config.depth, memory_dim], name="W_u_Q", initializer=self.initializer)
            W_u_P = tf.get_variable(shape=[self.config.depth, inputs_dim], name="W_u_P", initializer=self.initializer)
            # W_v_P = tf.get_variable(name="W_v_P", shape=[self.config.depth, self.config.dim])
            v = tf.get_variable(shape=[self.config.depth], name="v", initializer=self.initializer)

            W_u_Q_v = tf.tile(tf.expand_dims(tf.transpose(W_u_Q), axis=0),
                              multiples=[batch_size, 1, 1]) #[batch size, memory dim, depth]
            t1 = tf.matmul(memory, W_u_Q_v) #[batch, memory len, depth]
            t1 = tf.tile(tf.expand_dims(t1, axis=1), multiples=[1, i_len, 1, 1])
            W_u_P_v = tf.tile(tf.expand_dims(tf.transpose(W_u_P), axis=0),
                              multiples=[batch_size, 1, 1]) #[batch size, inputs_dim, depth]
            t2 = tf.matmul(inputs, W_u_P_v) #[batch, inputs len, depth]
            t2 = tf.tile(tf.expand_dims(t2, axis=2), multiples=[1, 1, m_len, 1]) #[batch, inputs len, memory len, depth]
            v_t = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(v, axis=1), axis=0), axis=0),
                          multiples=[batch_size, i_len, 1, 1])#[batch, inputs len, depth, 1]
            s = tf.matmul(tf.tanh(t1 + t2), v_t) #[batch size, inputs len, memory len, 1]
            s = tf.squeeze(s, axis=[3]) #[batch size, inputs len, memory len]
            mask = tf.sequence_mask(memory_len)
            # print("memory len shape ", memory_len.get_shape())
            # print("mask shape ", mask.get_shape())
            # print("s ", s.get_shape())
            mask = tf.tile(tf.expand_dims(mask, axis=1),
                           multiples=[1, i_len, 1])#[batch size, inputs len, memory len]

            if self.use_len:
                s = tf.where(mask, s, tf.fill(tf.shape(s), -float('inf')))

            a = tf.nn.softmax(s, axis=-1)

            if self.fetch_info is not None:
                self.fetch_info.add_info("%s_attn_softmax" %self.name, a)


            memory_t = tf.expand_dims(memory, axis=1)#[batch, 1, memory len, memory dim]
            c = tf.reduce_sum(tf.multiply(memory_t, tf.expand_dims(a, axis=-1)), axis=2) #[batch, inputs len, memory dim]

            new_inputs = tf.concat([inputs, c], axis=-1) #[batch, inputs len, new dim]


            new_inputs = self._add_gate(new_inputs, memory_dim + inputs_dim, batch_size)

            # if "cudnn" not in self.config.cell:
            #     outputs, state = tf.nn.dynamic_rnn(self._basic_cell,
            #                                        inputs=new_inputs,
            #                                        sequence_length=inputs_len,
            #                                        dtype=tf.float32)
            # else:
            #     outputs, state = self._basic_cell(new_inputs)



            #no cudnn
            rnn_encoder = RNNEncoder(self.config, "%s_rnn_encoding"%self.name)
            outputs = rnn_encoder(new_inputs, inputs_len)


            if self.config.dropout_type == "var":
                outputs = variational_rnn_dropout(outputs, self.config.dropout_keep_prob, time_rank=1)
            elif self.config.dropout_type == "ind":
                outputs = independent_dropout(outputs, self.config.dropout_keep_prob)
            elif self.config.dropout_type == "None":
                outputs = outputs

            return outputs


    def _add_gate(self, new_inputs, new_dim, batch_size):

        # if self.config.gated == "full_dim":
        #     W_gated = tf.get_variable(shape=[new_dim, new_dim], name="W_gated", initializer=self.initializer)
        #     W_gated_v = tf.tile(tf.expand_dims(W_gated, axis=0), multiples=[batch_size, 1, 1])
        #     gates = tf.nn.sigmoid(tf.matmul(new_inputs, W_gated_v))
        #     new_inputs = tf.multiply(new_inputs, gates)
        # elif self.config.gated == "one_gate_per_word":
        #     W_gated = tf.get_variable(shape=[new_dim, 1], name="W_gated", initializer=self.initializer)
        #     W_gated_v = tf.tile(tf.expand_dims(W_gated, axis=0), multiples=[batch_size, 1, 1])
        #     gates = tf.nn.sigmoid(tf.matmul(new_inputs, W_gated_v))
        #     new_inputs = tf.multiply(new_inputs, gates)
        # elif self.config.gated == "one_gate_per_word_v2":
        #
        #     W_gated = tf.get_variable(shape=[new_dim, 1], name="W_gated", initializer=self.initializer)
        #     W_gated_hidden = tf.get_variable(shape=[new_dim, new_dim], name="W_gated_hidden", initializer=self.initializer)
        #     W_gated_hidden_v = tf.tile(tf.expand_dims(W_gated_hidden, axis=0), multiples=[batch_size, 1, 1])
        #     W_gated_v = tf.tile(tf.expand_dims(W_gated, axis=0), multiples=[batch_size, 1, 1])
        #     gates = tf.matmul(tf.nn.tanh(tf.matmul(new_inputs, W_gated_hidden_v)), W_gated_v)
        #     gates = tf.nn.sigmoid(gates)
        #     new_inputs = tf.multiply(new_inputs, gates)
        if self.config.gated == "full_dim":
            gates = tf.layers.dense(new_inputs, units=new_dim, activation=tf.nn.sigmoid)
            new_inputs = tf.multiply(new_inputs, gates)
        elif self.config.gated == "one_gate_per_word":
            gates = tf.layers.dense(new_inputs, units=1, activation=tf.nn.sigmoid)
            new_inputs = tf.multiply(new_inputs, gates)
        elif self.config.gated == "one_gate_per_word_v2":
            gates = tf.layers.dense(new_inputs, units=new_dim, activation=tf.nn.tanh)
            # print("gate shape ", gates.get_shape())
            gates = tf.layers.dense(gates, units=1, activation=tf.nn.sigmoid)
            # print("gate shape ", gates.get_shape())
            new_inputs = tf.multiply(new_inputs, gates)
        elif self.config.gated == "None":
            new_inputs = new_inputs
        else:
            raise NotImplementedError()

        if self.config.gated != "None" and self.fetch_info is not None:
            # print(self.name)
            # exit(1)
            self.fetch_info.add_info("%s_attn_gate" %self.name, gates)

        return new_inputs






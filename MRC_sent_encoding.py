import tensorflow as tf
import numpy as np
from atten_no_wrapper import AttentionRNetNoWrapper
from rnn_encoding import RNNEncoder
import util as encoding_util



'''
attention
'''
class MRCSentsEncoder(object):

    def __init__(self, config, name, fetch_info=None, list_loader=None):
        self.name = name
        self.config = config
        self.fetch_info = fetch_info
        self.list_loader=list_loader

    def __call__(self, query, sents, query_len, sents_len):

        # sents = input_dic["sents_embedding"] #[batch, num_sent, max length, dim]
        # query = input_dic["query_embedding"]
        # old_shape = sents.get_shape().as_list()
        print("sentence encoding type ",self.config.attention.type)
        # if self.config.attention.type == "AttentionBahdanau":
        #     self.attention = attention.AttentionBahdanau(self.config.attention, self.name)
        # elif self.config.attention.type == "AttentionLuong":
        #     self.attention = attention.AttentionLuong(self.config.attention, self.name)
        if self.config.attention.type == "rnet":
            self.attention = AttentionRNetNoWrapper(self.config.attention, self.name, self.fetch_info)
        elif self.config.attention.type == "None":
            self.attention = None
        else:
            raise NotImplementedError

        return self.rnn_encoding(sents, query, query_len, sents_len)


    def _base_rnn_encoding(self, sents, sents_len):

        # new_sents = tf.reshape(sents, shape=[batch_size*max_sent_len, num_words, dim])
        # new_sents_len = tf.reshape(sents_len, shape=[batch_size*max_sent_len])

        #before doing attention, run rnn
        rnnencoder = RNNEncoder(config=self.config.base_rnn, name="base_rnn%s"%self.name)
        outputs = rnnencoder(sents, sents_len)
        return outputs

    def rnn_encoding(self, sents, query, query_len, sents_len):

        old_shape = tf.shape(sents)
        batch_size = old_shape[0]
        max_sent_len = old_shape[1]
        num_words = old_shape[2]
        dim = sents.get_shape()[3]
        # print(query.get_shape().as_list())
        # exit(1)


        query_shape = tf.shape(query)
        query_max_length = query_shape[1]
        query_dim = query.get_shape()[2]


        # sents = tf.concat(tf.unstack(sents, axis=1), axis=0)

        new_sents = tf.reshape(sents, shape=[batch_size*max_sent_len, num_words, dim])
        new_sents_len = tf.reshape(sents_len, shape=[batch_size*max_sent_len])
        query_as_memory = tf.expand_dims(query, axis=1) #[batch, 1, num words, dim]
        query_as_memory = tf.tile(query_as_memory, multiples=[1, max_sent_len, 1, 1]) #query: [batch, max_sent_len, num words, dim]
        query_as_memory = tf.reshape(query_as_memory, shape=[batch_size * max_sent_len,
                                                                 query_max_length,
                                                                 query_dim])

        # debug([tf.shape(new_sents), tf.shape(query_as_memory)], self.list_loader)

        query_len_as_memory = tf.expand_dims(query_len, axis=1)
        query_len_as_memory = tf.tile(query_len_as_memory, multiples=[1, max_sent_len])
        query_len_as_memory = tf.reshape(query_len_as_memory, shape=[batch_size * max_sent_len])
        # sents = tf.reshape(sents,
        #                    shape=tf.TensorShape([None, old_shape[2], old_shape[3]]),)



        if self.config.base_rnn.enable:
            if self.config.base_rnn.type == "rnn":
                new_sents = self._base_rnn_encoding(new_sents, new_sents_len)
                rnn_sents_output_dim = (self.config.base_rnn.dim*2) if self.config.base_rnn.direction == "bi" else self.config.base_rnn.dim
                sents = tf.reshape(new_sents, shape=[batch_size, max_sent_len, num_words, rnn_sents_output_dim])


        # print(query_as_memory.get_shape())
        # print(new_sents.get_shape())
        # print(query_len_as_memory.get_shape())
        # print(new_sents_len.get_shape())
        if self.config.attention.type == "None":
            outputs = sents
        else:
            attn_outputs = self.attention(memory=query_as_memory,
                                        inputs=new_sents,
                                        memory_len=query_len_as_memory,
                                        inputs_len=new_sents_len)
            #
            if self.config.attention.direction == "bi":
                output_dim = self.config.attention.dim * 2
            elif self.config.attention.direction == "forward":
                output_dim = self.config.attention.dim
            else:
                raise NotImplementedError

            attn_outputs = tf.reshape(attn_outputs, shape=[batch_size, max_sent_len, num_words, output_dim])
            outputs = encoding_util.residual_connection(sents, attn_outputs, self.config.attention.res)

            #debug([tf.shape(attn_outputs), tf.shape(outputs)], self.list_loader)

        return outputs

    def naive_encoding(self):
        pass

    def cnn_encoding(self):
        pass






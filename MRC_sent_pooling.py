import tensorflow as tf
from rnn_encoding import RNNEncoder

class MRCSentPooling():

    def __init__(self, config, name, fetch_info=None):
        self.name = name
        self.config = config
        self.use_len = False
        self.fetch_info = fetch_info


    def __call__(self, sents_encoding, sents_len):

        if self.config.type == "max_mean":
            chunk_encoding = self._max_mean(sents_encoding, sents_len)
        elif self.config.type == "conv":
            chunk_encoding = self._conv(sents_encoding, sents_len)
        elif self.config.type == "rnn":
            chunk_encoding = self._rnn(sents_encoding, sents_len)
        elif self.config.type == "mean":
            chunk_encoding = self._mean(sents_encoding, sents_len)
        else:
            raise NotImplementedError

        chunk_len = self._compute_chunk_len(sents_len)

        return chunk_encoding, chunk_len

    def _compute_chunk_len(self, sents_len):
        #sents_len [batch, num sents]
        sents_indicator = tf.where(sents_len > 0,
                                   tf.ones(shape=tf.shape(sents_len)),
                                   tf.zeros(shape=tf.shape(sents_len)))
        chunk_len = tf.cast(tf.reduce_sum(sents_indicator, -1), dtype=tf.int32)

        return chunk_len

    def _max_mean(self, sents_encoding, sents_len):

        sent_max = tf.reduce_max(sents_encoding, axis=2)

        if self.use_len:

            sents_len_for_mean = tf.cast(tf.expand_dims(sents_len, axis=-1), tf.float32)
            sents_len_for_mean = tf.tile(sents_len_for_mean, multiples=[1,1,sents_encoding.get_shape()[-1]])
            sent_sum = tf.reduce_sum(sents_encoding, axis=2)
            sent_mean = tf.where(tf.equal(sents_len_for_mean, 0),
                                 tf.zeros(tf.shape(sents_len_for_mean)),
                                 sent_sum / sents_len_for_mean)
        else:
            sent_mean = tf.reduce_mean(sents_encoding, axis=2)

        if self.fetch_info is not None:
            self.fetch_info.add_info("max_mean_sents_pooling_argmax", tf.argmax(sents_encoding, axis=2))

        # words_indices = tf.cast(tf.sequence_mask(sents_len), tf.float32)
        # words_indices = tf.tile(tf.expand_dims(words_indices, -1),
        #                         multiples=[1,1,1,sents_encoding.get_shape()[-1]])

        #
        # sent_mean = tf.reduce_mean(sents_encoding, axis=2)
        sents_encoding = tf.concat([sent_max, sent_mean], axis=-1)

        return sents_encoding

    def _mean(self, sents_encoding, sents_len):


        sent_mean = tf.reduce_mean(sents_encoding, axis=2)

        if self.fetch_info is not None:
            self.fetch_info.add_info("mean_sents_pooling_argmax", tf.argmax(sents_encoding, axis=2))

        # words_indices = tf.cast(tf.sequence_mask(sents_len), tf.float32)
        # words_indices = tf.tile(tf.expand_dims(words_indices, -1),
        #                         multiples=[1,1,1,sents_encoding.get_shape()[-1]])

        #
        # sent_mean = tf.reduce_mean(sents_encoding, axis=2)
        # sents_encoding = tf.concat([sent_max, sent_mean], axis=-1)

        return sent_mean


    def _conv(self, sents_encoding, sents_len):
        # old_shape = tf.shape(sents_encoding)
        # batch_size = old_shape[0]
        # max_sent_len = old_shape[1]
        # num_words = old_shape[2]
        dim = sents_encoding.get_shape()[3]
        filter_width = self.config.filter_width
        # new_sents = tf.reshape(sents_encoding, shape=[batch_size*max_sent_len, num_words, dim])
        with tf.variable_scope("conv%s"%self.name, reuse=tf.AUTO_REUSE):
            # filter = tf.get_variable(shape=[filter_width, dim, self.config.out_dim],
            #                          name="filter%s"%self.name,
            #                          initializer=tf.random_normal_initializer(stddev=0.5))

            filter = tf.get_variable(shape=[1, filter_width, dim, self.config.dim],
                                   name="filter%s"%self.name,
                                   initializer=tf.random_normal_initializer(stddev=0.5))
            # filter_2d = tf.tile(core, multiples=[1, max_sent_len, 1, 1]) #[width, max_sent_len, in dim, out dim]
            sents_encoding = tf.nn.conv2d(input=sents_encoding,
                                           filter=filter,
                                           strides=[1, 1, self.config.stride,1],
                                           padding="SAME")
            # outputs_width = tf.shape(sents_encoding)[2]
            # print("conv shape ", sents_encoding.get_shape())
            #
            # last_filter = tf.get_variable(shape=[1, 1, self.config.out_dim, self.config.out_dim],
            #                             name="last_filter%s"%self.name,
            #                             initializer=tf.random_normal_initializer(stddev=0.5))
            # last_filter = tf.tile(last_filter, multiples=[1,outputs_width,1,1])
            #
            # outputs = tf.nn.conv2d(input=sents_encoding,
            #                        filter=last_filter,
            #                        strides=[1, 1, outputs_width, 1],
            #                        padding="SAME")
            #
            # print("conv shape ", outputs.get_shape())
            #
            # outputs = tf.squeeze(outputs, axis=[2])
            # print("conv shape ", outputs.get_shape())

            outputs = tf.reduce_max(sents_encoding, axis=2)

        return outputs

    def _rnn(self, sents_encoding, sents_len):
        old_shape = tf.shape(sents_encoding)
        batch_size = old_shape[0]
        max_sent_len = old_shape[1]
        num_words = old_shape[2]
        dim = sents_encoding.get_shape()[3]

        # sents = tf.concat(tf.unstack(sents, axis=1), axis=0)

        new_sents = tf.reshape(sents_encoding, shape=[batch_size*max_sent_len, num_words, dim])
        new_sents_len = tf.reshape(sents_len, shape=[batch_size*max_sent_len])

        rnn_encoder = RNNEncoder(self.config, name="rnn%s"%self.name)
        outputs, state = rnn_encoder(new_sents, new_sents_len)

        print("state shape ", state.get_shape())

        if self.config.direction == "bi":
            state_dim = self.config.dim * 2
        else:
            state_dim = self.config.dim

        state = tf.reshape(state, shape=[batch_size, max_sent_len, state_dim])

        max_mean = self._max_mean(sents_encoding, sents_len)

        results = tf.concat([max_mean, state], axis=-1)

        return results




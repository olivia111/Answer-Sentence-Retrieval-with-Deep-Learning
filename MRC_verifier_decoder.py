import tensorflow as tf
from rnn_encoding import SeqPooling

'''
dim: hidden state size
q_emb_size: size of v
attn_size: size of matrix used to calculate attention

'''
class MRCVerifierDecoder:

    def __init__(self, config, name, fetch_info=None):
        self.config = config
        self.name = name
        self.fetch_info = fetch_info

    def __call__(self, probs, query_vector, chunk_encoding, chunk_len, query_len, fetch_info=None):

        return self.use_all(probs, query_vector, chunk_encoding, chunk_len, query_len)

    def use_all(self, probs, query_vector, chunk_encoding, chunk_len, query_len):

        no_answer_score = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # header_vector = tf.reduce_sum(tf.multiply(chunk_encoding, tf.expand_dims(probs[0], -1)), axis=1)
            vectors = [query_vector]
            if "mean" in self.config.type:
                for i in range(0, len(probs)):
                    vectors.append(tf.reduce_sum(tf.multiply(chunk_encoding, tf.expand_dims(probs[i], -1)), axis=1))

                if "mean" in self.config.type:
                    feature = tf.concat(vectors, axis=-1)
                    feature = tf.reshape(feature,
                                         shape=[-1,
                                                query_vector.get_shape()[-1] + len(probs) * chunk_encoding.get_shape()[
                                                    -1]])
                else:
                    raise NotImplementedError("MRC_verifier_decoder_use_all %s" % self.config.type)


                hidden_states = []
                for i in range(0, len(probs)):
                    h_state = tf.layers.dense(inputs=feature,
                                              units=self.config.layer1_dim,
                                              activation=tf.nn.relu,
                                              name="hidden_state%d" % i)
                    score = tf.layers.dense(inputs=h_state,
                                            units=1,
                                            name="no_answer_score%d" % i)
                    hidden_states.append(h_state)
                    no_answer_score.append(score)

            elif "header" in self.config.type:
                query_vector_dim = query_vector.get_shape()[-1]
                header_embedding = tf.reduce_sum(tf.multiply(chunk_encoding, tf.expand_dims(probs[0], -1)), axis=1)
                header_embedding_dim_align = tf.layers.dense(inputs=header_embedding,
                                                            units=query_vector_dim,
                                                            name="header_dimension_align")
                score = tf.reduce_sum(tf.multiply(query_vector, header_embedding_dim_align), axis=-1)
                score = tf.expand_dims(score, axis=-1)
                print("no answer_score dim")
                print(score.get_shape().as_list())

                for i in range(0, len(probs)):
                    no_answer_score.append(score)

            return no_answer_score







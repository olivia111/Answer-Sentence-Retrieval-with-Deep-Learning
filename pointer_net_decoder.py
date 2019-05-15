import tensorflow as tf
from rnn_encoding import SeqPooling
from pointer_net import MRCPointerNet

'''
dim: hidden state size
q_emb_size: size of v
attn_size: size of matrix used to calculate attention

'''
class MRCPointerNetDecoder(MRCPointerNet):


    def __call__(self, query_encoding, chunk_encoding, query_len, chunk_len):

        self.hidden_state_dim = query_encoding.get_shape().as_list()[-1]
        self.batch_size = tf.shape(query_encoding)[0]

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # outputs = self._query_pooling(query_encoding, query_len)
            basic_rnn_cell = self.get_basic_cell(name=self.config.cell)

            print("lstm state size ", basic_rnn_cell.state_size)
            #
            query_vector = self._query_pooling(query_encoding, query_len)
            cur_state = self.make_initial_state(query_vector, basic_rnn_cell)

            # if self.fetch_info is not None:
            #     self.fetch_info.add_info("%s_decoding_1_softmax" %self.name, a1)
            probs, best_indices, scores = [], [], []
            num_decoding_steps = self.config.num_steps

            for i in range(num_decoding_steps):
                a, p, score, state, output = self._one_step_decoding(cur_state, chunk_encoding, basic_rnn_cell, chunk_len)
                probs.append(a)
                best_indices.append(p)
                scores.append(score)
                cur_state = state

                if self.fetch_info is not None:
                    self.fetch_info.add_info("%s_decoding_%d_softmax" % (self.name, i), a)


            # if self.fetch_info is not None:
            #     self.fetch_info.add_info("%s_decoding_2_softmax" % self.name, a2)

            return probs, best_indices, scores, query_vector






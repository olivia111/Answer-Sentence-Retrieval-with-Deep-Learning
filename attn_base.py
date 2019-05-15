import tensorflow as tf
from abc import ABCMeta
from abc import abstractmethod
from var_dropout import variational_rnn_dropout, independent_dropout

'''
type: "AttentionBahdanau"
depth: 100
size: 100
dropout_keep_prob: 0.5
'''


class AttentionBase(metaclass=ABCMeta):

    def __init__(self, config, name):

        self.name = name
        self.config = config

        if not (config.type and config.depth and config.dim and config.dropout_keep_prob):
            raise Exception("not all config values are available %s" %name)


    @abstractmethod
    def __call__(self, memory, inputs, memory_len, inputs_len):
        pass

    @property
    def _basic_cell(self):
        print("this function is current diabled _basic_cell AttentionBase")
        basic_cell = None
        if self.config.cell == "lstm":
            basic_cell = tf.contrib.rnn.BasicLSTMCell(self.config.dim, name="cell%s"%self.name)
        elif self.config.cell == "gru":
            basic_cell = tf.contrib.rnn.GRUCell(self.config.dim, name="cell%s"%self.name)
        elif self.config.cell == "cudnnlstm":
            basic_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                        num_units=self.config.dim,
                                                        name="cell%s"%self.name,
                                                        kernel_initializer=tf.random_normal_initializer(stddev=0.5))
        elif self.config.cell == "cudnngru":
            basic_cell = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1,
                                                        num_units=self.config.dim,
                                                        name="cell%s"%self.name,
                                                        kernel_initializer=tf.random_normal_initializer(stddev=0.5))
        else:
            raise NotImplementedError

        # if "cudnn" not in self.config.cell and self.config.dropout_keep_prob < 1.0:
        #     basic_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=self.config.dropout_keep_prob)
        # elif "cudnn" in self.config.cell and self.config.dropout_keep_prob < 1.0:
        #     raise NotImplementedError("dropout for cudnn not implemented")

        return basic_cell


    def get_basic_cell(self, name):
        basic_cell = None
        if self.config.cell == "lstm":
            basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.dim, name=name)
        elif self.config.cell == "gru":
            basic_cell = tf.contrib.rnn.GRUCell(num_units=self.config.dim, name=name)
        else:
            raise NotImplementedError

        return basic_cell


# class AttentionBahdanau(AttentionBase):
#
#
#     def __init__(self, config, name):
#         super().__init__(config, name)
#
#
#     def __call__(self, memory, inputs, memory_len, inputs_len):
#
#         print("prining memory len")
#         # print(tf.rank(memory_len))
#
#         with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
#             attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.config.depth,
#                                                                   memory=memory,
#                                                                   memory_sequence_length=memory_len)
#             attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell=self._basic_cell,
#                                                             attention_mechanism=attn_mechanism,
#                                                             attention_layer_size=self.config.dim)
#
#             if "cudnn" not in self.config.cell:
#                 outputs, state = tf.nn.dynamic_rnn(attn_cell,
#                                                    inputs=inputs,
#                                                    sequence_length=inputs_len,
#                                                    dtype=tf.float32)
#             else:
#                 raise NotImplementedError("cudnn attention not implemented")
#
#
#             if self.config.dropout_type == "var":
#                 outputs = variational_rnn_dropout(outputs, self.config.drop_keep_prob, time_rank=1)
#             elif self.config.dropout_type == "ind":
#                 outputs = independent_dropout(outputs, self.config.drop_keep_prob)
#             elif self.config.dropout_type == "None":
#                 outputs = outputs
#
#
#
#         return outputs, state
#
#
# class AttentionLuong(AttentionBase):
#
#
#     def __init__(self, config, name):
#         super().__init__(config, name)
#
#
#
#     def __call__(self, memory, inputs, memory_len, inputs_len):
#
#         with tf.variable_scope(self.name):
#             attn_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.depth,
#                                                                memory,
#                                                                memory_sequence_length=memory_len)
#             attn_cell = tf.contrib.seq2seq.AttentionWrapper(self._basic_cell, attn_mechanism, self.config.dim)
#             outputs, state = tf.nn.dynamic_rnn(attn_cell, inputs,
#                                                sequence_length=inputs_len,
#                                                dtype=tf.float32)
#
#         if self.config.dropout_type == "var":
#             outputs = variational_rnn_dropout(outputs, self.config.drop_keep_prob, time_rank=1)
#         elif self.config.dropout_type == "ind":
#             outputs = independent_dropout(outputs, self.config.drop_keep_prob)
#         elif self.config.dropout_type == "None":
#             outputs = outputs
#
#         return outputs, state
#
#
# class AttentionRNet(AttentionBase):
#
#     #implementation: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf section 3.2
#     def __init__(self, config, name):
#         super(config, name)
#
#     def __call__(self, memory, inputs, memory_len, inputs_len):
#
#         with tf.variable_scope(self.name):
#             attn_cell = RNetAttnWrapper(memory, self.config, self._basic_cell)
#             outputs, state = tf.nn.dynamic_rnn(attn_cell, inputs,
#                                                sequence_length=inputs_len)
#         return outputs, state


















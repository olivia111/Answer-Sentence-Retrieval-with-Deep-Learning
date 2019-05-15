import tensorflow as tf
from var_dropout import variational_rnn_dropout, independent_dropout
from debug import debug
from abc import ABCMeta
from abc import abstractmethod


class SeqEncoderBase(metaclass=ABCMeta):

    def __init__(self, config, name):

        self.config = config
        self.name = name

        self.use_len = False

        if not (config.dim and config.dropout_keep_prob):
            raise Exception("not all config values exist %s" %name)

    @abstractmethod
    def __call__(self, inputs, inputs_len): #[None, max length, dim]

        pass



'''
direction
cell
dim
dropout_keep_prob
'''
class RNNEncoder(SeqEncoderBase):

    def __init__(self, config, name):

        super().__init__(config, name)
        if not (config.direction and config.cell):
            raise Exception("RNN encoder not all config values exist %s" %name)


    def get_basic_cell(self, name):
        basic_cell = None
        if self.config.cell == "lstm":
            basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.dim, name=name)
        elif self.config.cell == "gru":
            basic_cell = tf.contrib.rnn.GRUCell(num_units=self.config.dim, name=name)
        elif self.config.cell == "cudnnlstm":
            basic_cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1,
                                                        num_units=self.config.dim,
                                                        name=name)
        else:
            raise NotImplementedError

        # if "cudnn" not in self.config.cell and self.config.dropout_keep_prob < 1.0:
        #     basic_cell = tf.contrib.rnn.DropoutWrapper(basic_cell, output_keep_prob=self.config.dropout_keep_prob)
        # if "cudnn" in self.config.cell and self.config.dropout_keep_prob == 1.0:
        #     raise NotImplementedError("cudnn not implement yet")

        return basic_cell

    def get_initial_state(self, inputs, basic_cell):

        batch_size = tf.shape(inputs)[0]
        return basic_cell.zero_state(batch_size, dtype=tf.float32)

    def get_final_state(self, final_state):

        if self.config.cell == "lstm":
            return final_state.c
        elif self.config.cell == "gru":
            return final_state
        else:
            raise NotImplementedError("get final state")


    def __call__(self, inputs, inputs_len): #[None, max length, dim]

        print(inputs.get_shape().as_list())

        # mask = tf.expand_dims(tf.sequence_mask(inputs_len), -1)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            outputs, state = None, None
            if self.config.direction == "bi":
                fw_basic_cell = self.get_basic_cell(name="fw")
                bw_basic_cell = self.get_basic_cell(name="bw")
                init_state = self.get_initial_state(inputs, fw_basic_cell)
                # init_state_bw = self.get_initial_state(inputs, bw_basic_cell)
                bw_inputs = tf.reverse_sequence(input=inputs,
                                                seq_lengths=inputs_len,
                                                seq_axis=1,
                                                batch_axis=0)

                if "cudnn" not in self.config.cell:
                    fw_outputs, fw_state = tf.nn.dynamic_rnn(cell=fw_basic_cell,
                                                            sequence_length=inputs_len,
                                                            inputs=inputs,
                                                            dtype=tf.float32,
                                                            initial_state=init_state)
                    bw_outputs, bw_state = tf.nn.dynamic_rnn(cell=bw_basic_cell,
                                                             sequence_length=inputs_len,
                                                             inputs=bw_inputs,
                                                             dtype=tf.float32,
                                                             initial_state=init_state)
                    # outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_basic_cell,
                    #                                                  cell_bw=bw_basic_cell,
                    #                                                  sequence_length=inputs_len,
                    #                                                  inputs=inputs,
                    #                                                  dtype=tf.float32,
                    #                                                  initial_state_fw=init_state_fw,
                    #                                                  initial_state_bw=init_state_bw)
                else:
                    fw_outputs, fw_state = fw_basic_cell(inputs, init_state=init_state)
                    bw_outputs, bw_state = bw_basic_cell(inputs, init_state=init_state)

                # outputs = [fw_outputs, bw_outputs]
                # state = [fw_state, bw_state]


                # mask = tf.tile(mask, tf.TensorShape([1, 1, 2 * self.config.dim]))
                outputs = tf.concat([fw_outputs, bw_outputs], axis=-1)
                state = tf.concat([self.get_final_state(fw_state), self.get_final_state(bw_state)], axis=-1)

            elif self.config.direction == "forward":

                basic_cell = self.get_basic_cell(name="fw")
                init_state = self.get_initial_state(inputs, basic_cell)

                if "cudnn" not in self.config.cell:
                    outputs, state = tf.nn.dynamic_rnn(cell=basic_cell,
                                                       sequence_length=inputs_len,
                                                       inputs=inputs,
                                                       dtype=tf.float32,
                                                       initial_state=init_state)
                else:
                    outputs, state = basic_cell(inputs, init_state=init_state)
                    state = self.get_final_state(state)

                state = self.get_final_state(state)
                # mask = tf.tile(mask, tf.TensorShape([1, 1, self.config.dim]))

            # outputs = tf.where(mask, outputs, tf.zeros(shape=tf.shape(outputs)))

            if self.config.dropout_type == "var":
                print("drop out type is var")
                outputs = variational_rnn_dropout(outputs, self.config.dropout_keep_prob, time_rank=1)
            elif self.config.dropout_type == "ind":
                outputs = independent_dropout(outputs, self.config.dropout_keep_prob)
            elif self.config.dropout_type == "None":
                #no dropout
                outputs = outputs


            return outputs


class RNNEncoderMultiSents(RNNEncoder):

    pass

    # def get_initial_state(self, inputs):
    #
    # def __call__(self, inputs, inputs_len):




class SeqPooling():


    def __init__(self, config, name="seq_pooling"):

        self.config = config
        self.name = name


    def __call__(self, sequence):

        if self.config == "max_mean":

            seq_max = tf.reduce_max(sequence, -2)
            seq_mean = tf.reduce_mean(sequence, -2)
            return tf.concat([seq_max, seq_mean], -2)
        else:
            raise NotImplementedError





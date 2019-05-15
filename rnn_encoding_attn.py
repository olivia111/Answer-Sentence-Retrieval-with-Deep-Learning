
from atten_no_wrapper import AttentionRNetNoWrapper

'''
type: "AttentionBahdanau"
depth: 5
dim: 5
dropout_keep_prob: -1
cell: "lstm"
'''
class RNNEncoderWithAttention():

    def __init__(self, config, name, fetch_info=None):
        self.config = config
        self.name = name
        self.fetch_info = fetch_info



    def __call__(self, memory, inputs, memory_len, inputs_len):

        # sents = input_dic["sents_embedding"] #[batch, num_sent, max length, dim]
        # query = input_dic["query_embedding"]
        # old_shape = sents.get_shape().as_list()
        # if self.config.type == "AttentionBahdanau":
        #     self.attention = attention.AttentionBahdanau(self.config, self.name)
        # elif self.config.type == "AttentionLuong":
        #     self.attention = attention.AttentionLuong(self.config, self.name)
        if self.config.type == "rnet":
            self.attention = AttentionRNetNoWrapper(self.config, self.name, fetch_info=self.fetch_info)
        else:
            raise NotImplementedError

        # print(memory_len.get_shape())
        # print(memory.get_shape())
        # print(inputs.get_shape())
        # print(memory_len.get_shape())
        # print(inputs_len.get_shape())
        outputs = self.attention(memory=memory,
                                        inputs=inputs,
                                        memory_len=memory_len,
                                        inputs_len=inputs_len)
        return outputs

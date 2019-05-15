


class FetchInfoFromModel:

    def __init__(self, mode):

        self.mode = mode
        self.run_ops = {}

        self.create_maps()


    #this is shared by all instances
    def create_maps(self):

        self.all_modes = ["debug_instance"]
        self.maps = {}
        self.maps["debug_instance"] = ["sents_encoding_attn_softmax",
                                       "sents_encoding_attn_gate",
                                       "max_mean_sents_pooling_argmax",
                                       "qc_attention_attn_softmax",
                                       "qc_attention_attn_gate",
                                       "self_attention_attn_softmax",
                                       "self_attention_attn_gate",
                                       "pointer_decoding_1_softmax",
                                       "pointer_decoding_2_softmax"]

    def add_info(self, tensor_name, tensor_object):

        if tensor_name in self.maps[self.mode]:
            print("%s added" %tensor_name)
            self.run_ops[tensor_name] = tensor_object

    def get_run_ops(self):

        return self.run_ops
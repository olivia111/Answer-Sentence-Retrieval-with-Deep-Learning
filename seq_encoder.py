from rnn_encoding import RNNEncoder


def get_sequence_encoder(config, name):

    if not config.type:
        raise Exception("sequence encoder type not defined")

    if config.type == "rnn":
        return RNNEncoder(config, name)
    else:
        raise NotImplementedError()
    # elif config.type == "transf":
    #     return TransEncoder(config, name)
    # elif config.type == "cnn":
    #     return CNNEncoder(config, name)





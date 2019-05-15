import tensorflow as tf




def residual_connection(inputs, outputs, connect_type):
    if connect_type == "concat":
        outputs = tf.concat([outputs, inputs], axis=-1)
    elif connect_type == "add":
        # check dimension
        assert outputs.get_shape().as_list()[-1] == inputs.get_shape().as_list()[-1]
        outputs = outputs + inputs
    elif connect_type == "None":
        print("no residual connection")
    else:
        raise NotImplementedError("residual_connection not implemented")

    return outputs




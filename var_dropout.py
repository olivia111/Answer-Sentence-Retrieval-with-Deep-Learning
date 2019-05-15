import tensorflow as tf


def variational_rnn_dropout(rnn_outputs, keep_prob, time_rank=-2):

    noise_shape = tf.shape(rnn_outputs)
    noise_shape = tf.concat([noise_shape[0: time_rank], [1], noise_shape[time_rank + 1:]], axis=-1)
    return tf.nn.dropout(x=rnn_outputs, keep_prob=keep_prob, noise_shape=noise_shape)

def independent_dropout(rnn_outputs, keep_prob):

    noise_shape = tf.shape(rnn_outputs)
    return tf.nn.dropout(x=rnn_outputs, keep_prob=keep_prob, noise_shape=noise_shape)

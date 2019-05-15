import tensorflow as tf
from debug import debug



class MRCCharEmbedding:

    def __init__(self, config, name):
        self.config = config
        self.name = name


    def __call__(self, input_dic, list_loader):

        self.list_loader = list_loader

        query = input_dic["query2chars"]
        paragraph = input_dic["sents2chars"]

        num_chars = list_loader.get_num_chars()

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            char_weights = tf.get_variable(name="char_weight",
                                           shape=[num_chars, self.config.size],
                                           initializer=tf.random_normal_initializer(stddev=0.1),
                                           trainable=True)

            q_char_embedding = tf.nn.embedding_lookup(char_weights, query)
            sents_char_embedding = tf.nn.embedding_lookup(char_weights, paragraph)

            q_char_embedding, sents_char_embedding = self.convert_char2word_embedding(q_char_embedding,
                                                                                      sents_char_embedding)

            input_dic["query2chars_em"] = q_char_embedding
            input_dic["sents2chars_em"] = sents_char_embedding

        return input_dic

    def convert_char2word_embedding(self, q_char_embedding, sents_char_embedding):
        #convert char to word level embedding

        if self.config.pooling == "maxmeanmin":
            return self.max_mean_encoding(q_char_embedding), self.max_mean_encoding(sents_char_embedding)
        elif self.config.pooling == "fixed_size":
            return self.concat_fixed_size(q_char_embedding, 4), self.concat_fixed_size(sents_char_embedding, 5)
        else:
            raise NotImplementedError("convert_char2word_embedding")


    def max_mean_encoding(self, embedding):

        max_em = tf.reduce_max(embedding, axis=-2)
        # min_em = tf.reduce_min(embedding, axis=-2)
        mean_em = tf.reduce_mean(embedding, axis=-2)

        return tf.concat([max_em, mean_em], axis=-1)

    # def concat_fixed_size(self, embedding, rank):
    #
    #     fixed_size = 8
    #     old_shape = tf.shape(embedding)
    #     char_dim = self.config.size
    #     num_chars = old_shape[-2]
    #
    #     new_embedding = tf.reshape(embedding, shape=[-1, num_chars, char_dim])
    #     first_dim_new_embedding = tf.shape(new_embedding)[0]
    #
    #     new_embedding = tf.cond(num_chars < fixed_size,
    #                             lambda: tf.pad(new_embedding, paddings=[[0,0], [0, fixed_size - num_chars], [0,0]], constant_values=0),
    #                             lambda: new_embedding[:, :, 0:fixed_size])
    #     print("char_dim ", char_dim)
    #
    #     if rank == 4:
    #         new_embedding = tf.reshape(new_embedding, shape=[old_shape[0], old_shape[1], fixed_size, char_dim])
    #     elif rank == 5:
    #         new_embedding = tf.reshape(new_embedding, shape=[old_shape[0], old_shape[1], old_shape[2], fixed_size, char_dim])
    #     else:
    #         raise NotImplementedError()
    #     debug([tf.shape(new_embedding), old_shape], self.list_loader)
    #     return new_embedding


    def concat_fixed_size(self, embedding, rank):

        fixed_size = 8
        num_chars = tf.shape(embedding)[-2]
        if rank == 4:
            new_embedding = tf.cond(num_chars < fixed_size,
                                    lambda: tf.pad(embedding,
                                                   paddings=[[0, 0], [0,0], [0, fixed_size - num_chars], [0, 0]],
                                                   constant_values=0),
                                    lambda: embedding[:, :, 0:fixed_size, :])
        elif rank == 5:
            new_embedding = tf.cond(num_chars < fixed_size,
                                    lambda: tf.pad(embedding,
                                                   paddings=[[0, 0], [0,0], [0,0], [0, fixed_size - num_chars], [0, 0]],
                                                   constant_values=0),
                                    lambda: embedding[:, :, :, 0:fixed_size, :])
        else:
            raise NotImplementedError()

        combined_embedding = tf.concat(tf.split(new_embedding, num_or_size_splits=fixed_size, axis=-2), axis=-1)
        combined_embedding = tf.squeeze(combined_embedding, axis=-2)

        old_shape = tf.shape(embedding)
        if rank == 4:
            combined_embedding = tf.reshape(combined_embedding, shape=[old_shape[0], old_shape[1], fixed_size * self.config.size])
        elif rank == 5:
            combined_embedding = tf.reshape(combined_embedding,
                                            shape=[old_shape[0], old_shape[1], old_shape[2], fixed_size * self.config.size])
        else:
            raise NotImplementedError()

        #debug([tf.shape(combined_embedding), tf.shape(new_embedding)], self.list_loader)


        return combined_embedding





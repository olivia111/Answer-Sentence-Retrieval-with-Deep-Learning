import tensorflow as tf
# import tensorflow.contrib.lookup as lookup
import numpy as np
from debug import debug
from MRC_list_multi_option_loader import MRCListMultiOptionLoader

class MRCEmbedding(object):

    def __init__(self, config, name):
        self.name = name
        self.config = config

        if self.config.method == "glove":
            self.normal_word_voc = self._load_glove()

        self.num_html_tags = MRCListMultiOptionLoader.get_num_html_tags()
        self.num_special_tokens = MRCListMultiOptionLoader.get_num_special_tokens()


    def __call__(self, input_dic, list_loader):

        query = input_dic["query"]
        paragraph = input_dic["sents2words"]
        # passages = input_dic["title"]

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            special_word_voc_weights = tf.get_variable(name="special_token2vec",
                                                       shape=[self.num_special_tokens, self.config.size],
                                                       initializer=tf.random_normal_initializer(stddev=0.1),
                                                       trainable=True)

            normal_word_voc_weights = tf.get_variable(name="norm_word2vec",
                                                      initializer=self.normal_word_voc,
                                                      trainable=False)
            # special_word_voc_weights = tf.get_variable(name="special_word_voc", \
            #                                            shape=[self.num_special_tokens, self.config.word.size],\
            #                                            initializer=tf.zeros, \
            #                                            trainable=True)
            tags_voc_weights = tf.get_variable(name="html_tag2vec",
                                               shape=[self.num_html_tags, self.config.size],
                                               initializer=tf.random_normal_initializer(stddev=0.1),
                                               trainable=True)

            word_voc_weights = tf.concat([special_word_voc_weights, normal_word_voc_weights, tags_voc_weights], axis=0)

            q_embedding = tf.nn.embedding_lookup(word_voc_weights, ids=query)
            p_embedding = tf.nn.embedding_lookup(word_voc_weights, ids=paragraph)

            input_dic["query_embedding"] = q_embedding
            input_dic["sents_embedding"] = p_embedding
            input_dic["init_sents_embedding"] = p_embedding

            # debug([tf.shape(p_embedding)], list_loader)

        return input_dic



    def _load_glove(self):

        word_voc = []
        words = []
        # with open(self.config.word.path, 'r') as f:
        with tf.gfile.GFile(self.config.path) as f:
            for line in f:
                segs = line.split(" ")
                w = segs[0]
                words.append(w)
                word_voc.append(list(map(float, segs[1:])))

        word_voc = np.array(word_voc, np.float32)

        assert self.config.size == word_voc.shape[1]

        return word_voc

    def _load_elmo(self):
        pass




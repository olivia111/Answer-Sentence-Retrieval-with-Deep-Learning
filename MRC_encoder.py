import tensorflow as tf
# import numpy as np
from MRC_embedding import MRCEmbedding
from MRC_sent_encoding import MRCSentsEncoder
from MRC_sent_pooling import MRCSentPooling
from rnn_encoding_attn import RNNEncoderWithAttention
from pointer_net_decoder import MRCPointerNetDecoder
from MRC_verifier_decoder import MRCVerifierDecoder
from MRC_char_embedding import MRCCharEmbedding
# from debug import debug
import util as encoding_util
from seq_encoder import get_sequence_encoder

class MRCEncoder(object):

    def __init__(self, config, global_config, name="MRCSentEncoding", fetch_info=None):
        self.name = name
        self.config = config
        self.global_config = global_config
        self.fetch_info = fetch_info
        self.enable_summary = self.global_config["enable_summary"]


    def build_encodings(self, input_dic, list_loader):

        #debug([input_dic["header_index"], input_dic["sents_len"], tf.shape(input_dic["sents_len"])] , list_loader)
        word_embedding = MRCEmbedding(self.config.embedding.word, name="word_embedding")
        input_dic = word_embedding(input_dic, list_loader)

        #debug([tf.shape(input_dic["sents_embedding"]), tf.shape(input_dic["sents2words"])], list_loader)

        if self.config.embedding.enable_char:
            print("char level embedding...")
            char_embedding = MRCCharEmbedding(self.config.embedding.char, name="char_embedding")
            input_dic = char_embedding(input_dic, list_loader)
            #concate char and word
            #debug([tf.shape(input_dic["query_embedding"]), tf.shape(input_dic["query2chars_em"]), tf.shape(input_dic["sents2chars_em"])], list_loader)
            input_dic["query_embedding"] = tf.concat([input_dic["query_embedding"],
                                                      input_dic["query2chars_em"]],
                                                     axis=-1)
            input_dic["sents_embedding"] = tf.concat([input_dic["sents_embedding"],
                                                      input_dic["sents2chars_em"]],
                                                     axis=-1)
        # debug([tf.shape(input_dic["query_embedding"])], list_loader)
        #debug([tf.shape(input_dic["query_embedding"]), tf.shape(input_dic["sents_embedding"])], list_loader)

        #encoding query
        print("encoding query")
        # q_encoder = RNNEncoder(self.config.query_encoding, name="query_encoding")
        #self.config.query_encoding.type == "cnn"
        q_encoder = get_sequence_encoder(self.config.query_encoding, name="query_encoding")
        outputs = q_encoder(input_dic["query_embedding"], input_dic["query_len"])
        input_dic["query_embedding"] = outputs

        # debug([tf.shape(outputs)], list_loader)

        #encoding sentences
        print("encoding sentences")
        # print(input_dic["query_len"].get_shape())
        sents_encoder = MRCSentsEncoder(self.config.sents_encoding, name="sents_encoding", fetch_info=self.fetch_info, list_loader=list_loader)
        outputs = sents_encoder(input_dic["query_embedding"],
                                input_dic["sents_embedding"],
                                input_dic["query_len"],
                                input_dic["sents_len"])

        input_dic["sents_encoding"] = outputs

        # debug([tf.shape(outputs)], list_loader)

        #sents pooling.
        #input_dic["sents_encoding"]: [batch_size, num sents, sent length, dim]
        print("sents pooling")
        sent_pooling = MRCSentPooling(self.config.sents_encoding.pooling, name="max_mean", fetch_info=self.fetch_info)
        chunk_encoding, chunk_len = sent_pooling(input_dic["sents_encoding"], input_dic["sents_len"])
        input_dic["chunk_encoding"] = chunk_encoding
        input_dic["chunk_len"] = chunk_len

        #debug([tf.shape(input_dic["sents_encoding"])], list_loader)

        # return input_dic

        # if enable_summary:
        #     with tf.name_scope("computable_summaries"):
        #         tf.summary.histogram("sents_encoding", input_dic["sents_encoding"])
        #         tf.summary.histogram("query_embedding", input_dic["query_embedding"])
        #         tf.summary.histogram("chunk_encoding_after_pooling", chunk_encoding)

        #RNN and self attention
        print("rnn with chunk")
        # rnn_encoder = RNNEncoder(self.config.chunk_encoding.rnn, name="chunk_encoding")
        initial_chunk_encoder = get_sequence_encoder(self.config.chunk_encoding.rnn, name="chunk_encoding")
        outputs = initial_chunk_encoder(input_dic["chunk_encoding"], input_dic["chunk_len"])
        outputs = encoding_util.residual_connection(input_dic["chunk_encoding"], outputs, self.config.chunk_encoding.rnn.res)
        input_dic["chunk_encoding"] = outputs
        #attentnion with query

        #debug([tf.shape(input_dic["chunk_encoding"])], list_loader)
        if self.config.chunk_encoding.q_match.enable:
            print("attention with query")
            outputs = RNNEncoderWithAttention(self.config.chunk_encoding.q_match,
                                              name="qc_attention",
                                              fetch_info=self.fetch_info)(memory=input_dic["query_embedding"],
                                                                           inputs=input_dic["chunk_encoding"],
                                                                           memory_len=input_dic["query_len"],
                                                                           inputs_len=input_dic["chunk_len"])

            outputs = encoding_util.residual_connection(input_dic["chunk_encoding"], outputs, self.config.chunk_encoding.q_match.res)

            input_dic["chunk_encoding"] = outputs

        # debug([tf.shape(input_dic["chunk_encoding"])], list_loader)

        #self attention
        if self.config.chunk_encoding.self_match.enable:
            print("self attention")
            outputs = RNNEncoderWithAttention(self.config.chunk_encoding.self_match,
                                              name="self_attention",
                                              fetch_info=self.fetch_info)(memory=input_dic["chunk_encoding"],
                                                                         inputs=input_dic["chunk_encoding"],
                                                                         memory_len=input_dic["chunk_len"],
                                                                         inputs_len=input_dic["chunk_len"])

            outputs = encoding_util.residual_connection(input_dic["chunk_encoding"], outputs, self.config.chunk_encoding.q_match.res)

        input_dic["chunk_encoding"] = outputs

        # debug([tf.shape(input_dic["chunk_encoding"])], list_loader)

        return input_dic

    def build_decision_layers(self, input_dic, list_loader):

        # if enable_summary:
        #     with tf.name_scope("computable_summaries"):
        #         tf.summary.histogram("pointer_net_inputs_c", input_dic["chunk_encoding"])
        #         tf.summary.histogram("pointer_net_inputs_q", input_dic["query_embedding"])
        #debug([tf.shape(input_dic["query_embedding"]), tf.shape(input_dic["chunk_encoding"])], list_loader)
        #pointer net
        print("pointer network")
        probs, best_indices, scores, q_vector = MRCPointerNetDecoder(self.config.pointer,
                                                         name="pointer",
                                                         fetch_info=self.fetch_info)(query_encoding=input_dic["query_embedding"],
                                                                                    chunk_encoding=input_dic["chunk_encoding"],
                                                                                    query_len=input_dic["query_len"],
                                                                                    chunk_len=input_dic["chunk_len"])
        input_dic["q_vector"] = q_vector

        input_dic["prob_header"] = probs[0]
        input_dic["p1"] = best_indices[0]
        input_dic["header_scores"] = scores[0]

        input_dic["prob_fb"] = probs[1]
        input_dic["p2"] = best_indices[1]
        input_dic["fb_scores"] = scores[1]

        #debug([input_dic["prob_header"], input_dic["header_scores"], input_dic["prob_fb"], input_dic["fb_scores"]] , list_loader)

        if self.config.pointer.num_steps > 2:
            input_dic["prob_sb"] = probs[2]
            input_dic["p3"] = best_indices[2]
            input_dic["sb_scores"] = scores[2]

        if self.config.pointer.num_steps > 3:
            input_dic["prob_tb"] = probs[3]
            input_dic["p4"] = best_indices[3]
            input_dic["tb_scores"] = scores[3]

        #verifier
        print("verifier")
        no_answer_scores = MRCVerifierDecoder(self.config.verifier,
                                              name="verifier",
                                              fetch_info=self.fetch_info)(probs,
                                                query_vector=input_dic["q_vector"],
                                                chunk_encoding=input_dic["chunk_encoding"],
                                                chunk_len=input_dic["chunk_len"],
                                                query_len=input_dic["query_len"])

        # input_dic["header_no_score"] = h_no_score
        # input_dic["fb_no_score"] = fb_no_score

        if self.config.verifier.enable:
            input_dic = self._adjust_probability(input_dic, no_answer_scores)
        else:
            input_dic = self._not_adjust_probability(input_dic)

        #debug([input_dic["adjust_prob_header"], input_dic["adjust_header_scores"]] , list_loader)

        # debug([p1, tf.shape(p1)], list_loader)
        return input_dic

    def __call__(self, input_dic, list_loader):
        input_dic = self.build_encodings(input_dic, list_loader)
        input_dic = self.build_decision_layers(input_dic, list_loader)
        return input_dic


    def _not_adjust_probability(self, input_dic):
        #just rename
        input_dic["adjust_prob_header"] = input_dic["prob_header"]
        input_dic["adjust_prob_fb"] = input_dic["prob_fb"]
        input_dic["adjust_p1"] = input_dic["p1"]
        input_dic["adjust_p2"] = input_dic["p2"]
        input_dic["adjust_header_scores"] = input_dic["header_scores"]
        input_dic["adjust_fb_scores"] = input_dic["fb_scores"]

        return input_dic


    def _adjust_probability(self, input_dic, no_answer_scores):
        #combine no score
        #in future, it will move a class
        header_scores = tf.concat([input_dic["header_scores"], no_answer_scores[0]], axis=-1)
        fb_scores = tf.concat([input_dic["fb_scores"], no_answer_scores[1]], axis=-1)
        h_probs = tf.nn.softmax(header_scores, axis=-1)
        fb_probs = tf.nn.softmax(fb_scores, axis=-1)


        # if self.config.decision_layer.type == "bias":
        #     header_scores, fb_scores = self._add_sample_bias(header_scores, fb_scores)
        # elif self.config.decision_layer.type == "dense":
        #     raise NotImplementedError("decision layer dense")
        # elif self.config.decision_layer.type == "None":
        #     pass

        input_dic["adjust_prob_header"] = h_probs
        input_dic["adjust_prob_fb"] = fb_probs
        input_dic["adjust_p1"] = tf.argmax(header_scores, -1)
        input_dic["adjust_p2"] = tf.argmax(fb_scores, -1)
        input_dic["adjust_header_scores"] = header_scores
        input_dic["adjust_fb_scores"] = fb_scores

        if len(no_answer_scores) > 2:
            sb_scores = tf.concat([input_dic["sb_scores"], no_answer_scores[2]], axis=-1)
            sb_probs = tf.nn.softmax(sb_scores, axis=-1)
            input_dic["adjust_prob_sb"] = sb_probs
            input_dic["adjust_p3"] = tf.argmax(sb_scores, -1)
            input_dic["adjust_sb_scores"] = sb_scores

        if len(no_answer_scores) > 3:
            tb_scores = tf.concat([input_dic["tb_scores"], no_answer_scores[2]], axis=-1)
            tb_probs = tf.nn.softmax(tb_scores, axis=-1)
            input_dic["adjust_prob_sb"] = tb_probs
            input_dic["adjust_p3"] = tf.argmax(tb_scores, -1)
            input_dic["adjust_sb_scores"] = tb_scores

        #debug([header_scores, input_dic["header_index"], fb_scores, input_dic["fb_index"]], self.list_loader)
        return input_dic



    def _add_sample_bias(self, header_scores, fb_scores):

        with tf.variable_scope(name_or_scope="sample_bias", reuse=tf.AUTO_REUSE):

            h_pos = tf.get_variable(shape=[1], name="header_pos_bias",
                                     initializer=tf.random_normal_initializer(stddev=0.5),
                                     trainable=True)
            h_neg = tf.get_variable(shape=[1], name="header_neg_bias",
                                    initializer=tf.random_normal_initializer(stddev=0.5),
                                    trainable=True)
            bullet_pos = tf.get_variable(shape=[1], name="bullet_pos_bias",
                                          initializer=tf.random_normal_initializer(stddev=0.5),
                                          trainable=True)
            bullet_neg = tf.get_variable(shape=[1], name="bullet_neg_bias",
                                         initializer=tf.random_normal_initializer(stddev=0.5),
                                         trainable=True)
            batch_size = tf.shape(header_scores)[0]
            num_sents = tf.shape(header_scores)[1] - 1
            print("header scores shape", header_scores.get_shape())
            print("fb scores shape", fb_scores.get_shape())
            h_pos_b = tf.tile(tf.expand_dims(h_pos, axis=1), multiples=[batch_size, num_sents])
            h_neg_b = tf.tile(tf.expand_dims(h_neg, axis=1), multiples=[batch_size, 1])
            bullet_pos_b = tf.tile(tf.expand_dims(bullet_pos, axis=1), multiples=[batch_size, num_sents])
            bullet_neg_b = tf.tile(tf.expand_dims(bullet_neg, axis=1), multiples=[batch_size, 1])

            header_scores = header_scores + tf.concat([h_pos_b, h_neg_b], axis=-1)
            fb_scores = fb_scores + tf.concat([bullet_pos_b, bullet_neg_b], axis=-1)

        return header_scores, fb_scores







import tensorflow as tf
from MRC_model import MRCModel


class MRCMultiOptionModel(MRCModel):

    def __call__(self, input_dic, list_loader):
        input_dic = self.build_graph(input_dic, list_loader)
        self.list_loader = list_loader
        input_dic = self._compute_loss_v2(input_dic)
        # input_dic = self._compute_eval_tensors(input_dic)
        # input_dic = self._compute_debug_info(input_dic)

        return input_dic



    def _compute_loss_v2(self, input_dic):

        loss_weights = self.config.loss.weights
        input_dic = self._compute_loss_multi_option(input_dic, loss_weights)
        return input_dic

    def _compute_loss_multi_option(self, input_dic, loss_weights):
        actual_batch_size = tf.shape(input_dic["raw_sents"])[0]
        indices_tensor = tf.expand_dims(tf.range(actual_batch_size, delta=1, dtype=tf.int32), axis=1)  # [batch size, 1]
        num_sents = tf.shape(input_dic["sents_len"])[1]

        # compute loss for header or bullets
        def _compute_loss_for_indices(indices, adjust_prob, lengths):
            # indices [batch size, number indices]
            # header_index = input_dic["header_index"]  #
            max_num_indices = tf.shape(indices)[1]
            batch_indices = tf.tile(indices_tensor, multiples=[1, max_num_indices])
            # [batch size, number indices, 2]
            index_to_gather = tf.concat([tf.expand_dims(batch_indices, axis=-1), tf.expand_dims(indices, axis=-1)],
                                        axis=-1)
            # return index_to_gather

            # header_index_to_gather = get_gather_nd_indices(input_dic["header_index"])
            # fb_index_to_gather = get_gather_nd_indices(input_dic[""])
            #debug([batch_indices, indices, adjust_prob, index_to_gather, lengths], self.list_loader)
            probs = tf.gather_nd(adjust_prob, indices=index_to_gather)

            #debug([probs, index_to_gather], self.list_loader)
            # fill 1 for out of range indices
            mask = tf.sequence_mask(lengths)
            probs = tf.where(mask, probs, tf.fill(tf.shape(probs), 0.0))

            # debug([probs, mask], self.list_loader)

            probs_sum = tf.reduce_sum(probs, axis=-1) + 0.000000001
            logits = -tf.log(probs_sum) #[batch size]

            # total_num_indices = tf.reduce_sum(lengths)
            logits_mean = tf.reduce_mean(logits)

            # debug([logits, probs_sum, logits_mean], self.list_loader)

            return logits_mean, probs, probs_sum

        def _convert_no_answer_option(indices):
            no_answer_num_input = -10000
            # header_index = batch["header_index"]
            # fb_index = batch["fb_index"]
            # max_num_sents = tf.shape(output["sents_len"])[1]
            indices_shape = tf.shape(indices)
            no_answer_num = tf.fill(dims=indices_shape, value=no_answer_num_input)
            # neg_one = tf.negative(tf.ones(shape=indices_shape, dtype=tf.int32))
            no_answer_index = tf.fill(dims=indices_shape, value=num_sents)
            new_indices = tf.where(tf.equal(indices, no_answer_num),
                                   no_answer_index,
                                   indices)

            return new_indices

        assert self.config.pointer.num_steps <= len(self.config.loss.weights)

        #since I already convert -1 to the number of sents. This is removed

        h_index = _convert_no_answer_option(input_dic["header_index"])
        fb_index = _convert_no_answer_option(input_dic["fb_index"])

        #
        # debug([h_index, input_dic["adjust_prob_header"], input_dic["header_length"]], self.list_loader)

        h_log_mean, h_probs, h_probs_sum = _compute_loss_for_indices(h_index, input_dic["adjust_prob_header"],
                                                                      input_dic["header_length"])


        fb_log_mean, fb_probs, fb_probs_sum = _compute_loss_for_indices(fb_index, input_dic["adjust_prob_fb"],
                                                                         input_dic["fb_length"])
        # input_dic["gather_header_prob"] = h_probs
        # input_dic["gather_fb_prob"] = fb_probs

        loss = h_log_mean * loss_weights[0] + fb_log_mean * loss_weights[1]

        if self.config.pointer.num_steps > 2:
            sb_index = _convert_no_answer_option(input_dic["sb_index"])
            sb_log_mean, sb_probs, sb_probs_sum = _compute_loss_for_indices(sb_index, input_dic["adjust_prob_sb"],
                                                                        input_dic["sb_length"])
            loss = loss + sb_log_mean * loss_weights[2]
            # input_dic["gather_sb_prob"] = sb_probs

        if self.config.pointer.num_steps > 3:
            tb_index = _convert_no_answer_option(input_dic["tb_index"])
            tb_log_mean, tb_probs, tb_probs_sum = _compute_loss_for_indices(tb_index, input_dic["adjust_prob_tb"],
                                                                        input_dic["tb_length"])
            loss = loss + tb_log_mean * loss_weights[3]
            # input_dic["gather_sb_prob"] = tb_probs

        # debug([loss], self.list_loader)
        # tb_loss, tb_probs = _compute_loss_for_indices(input_dic["tb_index"], input_dic["adjust_prob_tb"], input_dic["tb_length"])

        loss = tf.divide(loss, self.config.pointer.num_steps)

        # loss = tf.reduce_mean(-tf.log(tf.concat([prob_header, prob_fb], axis=-1)))
        # input_dic["debug_label_probs"] = tf.concat([prob_header, prob_fb], axis=-1)
        # input_dic["debug_header_scores"] = header_scores
        # input_dic["debug_fb_scores"] = fb_scores

        lambd = 0.005
        l2 = lambd * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        loss = loss + l2

        input_dic["loss"] = loss

        return input_dic



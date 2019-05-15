import tensorflow as tf

from MRC_encoder import MRCEncoder


class MRCModel():

    def __init__(self, config, global_config, name="MRCModel", fetch_info=None):
        self.name = name
        self.config = config
        self.global_config = global_config
        self.fetch_info = fetch_info


    def __call__(self, input_dic, list_loader):
        input_dic = self.build_graph(input_dic, list_loader)
        self.list_loader = list_loader
        # return input_dic
        input_dic = self._compute_loss_v2(input_dic)
        input_dic = self._compute_eval_tensors(input_dic)
        input_dic = self._compute_debug_info(input_dic)

        return input_dic


    def build_graph(self, input_dic, list_loader):
        # word embedding
        # input_dic = MRC_embedding(self.config.embedding)(input_dic)
        input_dic = MRCEncoder(self.config, self.global_config, name="MRCEncoder", fetch_info=self.fetch_info)(input_dic, list_loader)

        #no answer option?
        return input_dic

    def _compute_loss_no_verifier(self, input_dic):

        #to be deprecated
        h_probs = tf.nn.softmax(input_dic["header_scores"], axis=-1)
        fb_probs = tf.nn.softmax(input_dic["fb_scores"], axis=-1)
        input_dic["combined_h_prob"] = h_probs
        input_dic["combined_fb_prob"] = fb_probs

        #debug([header_scores, input_dic["header_index"], fb_scores, input_dic["fb_index"]], self.list_loader)


        prob_header = tf.gather_nd(h_probs, indices=input_dic["header_index"])
        prob_fb = tf.gather_nd(fb_probs, indices=input_dic["fb_index"])
        # print("prob_fb shape ", prob_fb.get_shape())
        # exit(1)
        loss = tf.reduce_mean(-tf.log(prob_header) - tf.log(prob_fb))
        input_dic["debug_label_probs"] = tf.concat([prob_header, prob_fb], axis=-1)
        input_dic["debug_header_scores"] = input_dic["header_scores"]
        input_dic["debug_fb_scores"] = input_dic["fb_scores"]
        input_dic["gather_header_prob"] = prob_header
        input_dic["gather_fb_prob"] = prob_fb


        input_dic["loss_with_answer"] = loss
        input_dic = self._combine_loss(input_dic)

        # debug([input_dic["loss"]], self.list_loader)

        return input_dic

    def _compute_loss_v2(self, input_dic):

        prob_header = tf.gather_nd(input_dic["adjust_prob_header"], indices=input_dic["header_index"])
        prob_fb = tf.gather_nd(input_dic["adjust_prob_fb"], indices=input_dic["fb_index"])

        #amplify fb loss
        header_loss = tf.reduce_mean(-tf.log(prob_header))
        fb_loss = tf.reduce_mean(-tf.log(prob_fb))
        loss = (header_loss + 5 * fb_loss) / 2

        #loss = tf.reduce_mean(-tf.log(tf.concat([prob_header, prob_fb], axis=-1)))
        input_dic["debug_label_probs"] = tf.concat([prob_header, prob_fb], axis=-1)
        # input_dic["debug_header_scores"] = header_scores
        # input_dic["debug_fb_scores"] = fb_scores
        input_dic["gather_header_prob"] = prob_header
        input_dic["gather_fb_prob"] = prob_fb

        input_dic["loss"] = loss


        return input_dic

    def _compute_eval_tensors(self, input_dic):

        # debug([input_dic["adjust_p1"], input_dic["header_index"]], self.list_loader)
        #compute EM
        #explan:
        #adjust_p1: the position of header that has the largest probability
        #adjust_p2: the position of fb that has the largest probability
        #adjust_prob_header: header probability
        #adjust_prob_fb: fb probability
        print("h_index ", (input_dic["header_index"][:,1]).get_shape())
        h_index = input_dic["header_index"][:, 1]
        fb_index = input_dic["fb_index"][:, 1]

        batch_size = tf.shape(input_dic["raw_sents"])[0]
        batch_index = tf.expand_dims(tf.range(batch_size, delta=1, dtype=tf.int32), axis=1)
        best_h_index = tf.concat([batch_index, tf.cast(tf.expand_dims(input_dic["adjust_p1"], axis=1), tf.int32)], axis=-1)
        best_fb_index = tf.concat([batch_index, tf.cast(tf.expand_dims(input_dic["adjust_p2"], axis=1), tf.int32)], axis=-1)

        # print(h_index.get_shape())
        # print(input_dic["adjust_p1"].get_shape())
        # debug([input_dic["adjust_p1"]], self.list_loader)
        # exit(1)

        # compute the probability of best header and first bullet
        # print(input_dic["adjust_prob_header"].get_shape())
        # print(input_dic["adjust_p1"].get_shape())
        # exit(1)


        input_dic["best_header_prob"] = tf.gather_nd(input_dic["adjust_prob_header"], indices=best_h_index)
        input_dic["best_fb_prob"] = tf.gather_nd(input_dic["adjust_prob_fb"], indices=best_fb_index)

        #compute entropy TO DO

        h_mask = tf.equal(tf.cast(input_dic["adjust_p1"], tf.int32), h_index)
        fb_mask = tf.equal(tf.cast(input_dic["adjust_p2"], tf.int32), fb_index)
        h_pred_r = tf.where(h_mask,
                            tf.ones(shape=tf.shape(h_index), dtype=tf.int32),
                            tf.zeros(shape=tf.shape(h_index), dtype=tf.int32))
        fb_pred_r = tf.where(fb_mask,
                             tf.ones(shape=tf.shape(fb_index), dtype=tf.int32),
                             tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))

        both_pred_r = tf.where(tf.logical_and(h_mask, fb_mask),
                               tf.ones(shape=tf.shape(h_index), dtype=tf.int32),
                               tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))

        num_correct_h = tf.reduce_sum(h_pred_r)
        num_correct_fb = tf.reduce_sum(fb_pred_r)
        num_correct_both = tf.reduce_sum(both_pred_r)

        input_dic["num_correct_header"] = num_correct_h
        input_dic["num_correct_fb"] = num_correct_fb
        input_dic["num_correct_both"] = num_correct_both

        batch_size = tf.shape(h_pred_r)[0]

        batch_h_p = num_correct_h / batch_size
        batch_fb_p = num_correct_fb / batch_size
        batch_both_p = num_correct_both / batch_size

        input_dic["header_em"] = batch_h_p
        input_dic["fb_em"] = batch_fb_p
        input_dic["both_em"] = batch_both_p

        gt_mask_fb = tf.less(fb_index, tf.ones(shape=tf.shape(fb_index), dtype=tf.int32) * tf.shape(input_dic["raw_sents"])[1])
        gt_mask_h = tf.less(h_index,
                          tf.ones(shape=tf.shape(fb_index), dtype=tf.int32) * tf.shape(input_dic["raw_sents"])[1])
        gt_mask = tf.logical_and(gt_mask_fb, gt_mask_h)
        gt_fb = tf.where(gt_mask_fb,
                           tf.ones(shape=tf.shape(fb_index), dtype=tf.int32),
                           tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))
        gt_h = tf.where(gt_mask_h,
                      tf.ones(shape=tf.shape(fb_index), dtype=tf.int32),
                      tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))
        gt = tf.where(gt_mask,
                      tf.ones(shape=tf.shape(fb_index), dtype=tf.int32),
                      tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))
        num_gt_fb = tf.reduce_sum(gt_fb)
        num_gt_h = tf.reduce_sum(gt_h)
        num_gt = tf.reduce_sum(gt)

        #compute precision of header and first bullet
        triggered_mask_fb = tf.less(tf.cast(input_dic["adjust_p2"], tf.int32), tf.ones(shape=tf.shape(fb_index), dtype=tf.int32) * tf.shape(input_dic["raw_sents"])[1])
        triggered_mask_h = tf.less(tf.cast(input_dic["adjust_p1"], tf.int32),
                                    tf.ones(shape=tf.shape(fb_index), dtype=tf.int32) *
                                    tf.shape(input_dic["raw_sents"])[1])
        triggered_fb = tf.where(triggered_mask_fb,
                             tf.ones(shape=tf.shape(fb_index), dtype=tf.int32),
                             tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))
        triggered_h = tf.where(triggered_mask_h,
                             tf.ones(shape=tf.shape(fb_index), dtype=tf.int32),
                             tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))
        triggered = tf.where(tf.logical_and(triggered_mask_fb, triggered_mask_h),
                             tf.ones(shape=tf.shape(fb_index), dtype=tf.int32),
                             tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))
        num_triggered_fb = tf.reduce_sum(triggered_fb)
        num_triggered_h = tf.reduce_sum(triggered_h)
        num_triggered = tf.reduce_sum(triggered)

        #
        correct_triggered_fb = tf.where(tf.logical_and(gt_mask_fb, fb_mask),
                                     tf.ones(shape=tf.shape(h_index), dtype=tf.int32),
                                     tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))
        correct_triggered_h = tf.where(tf.logical_and(gt_mask_h, h_mask),
                                     tf.ones(shape=tf.shape(h_index), dtype=tf.int32),
                                     tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))
        correct_triggered = tf.where(tf.logical_and(gt_mask, tf.logical_and(h_mask, fb_mask)),
                                     tf.ones(shape=tf.shape(h_index), dtype=tf.int32),
                                     tf.zeros(shape=tf.shape(fb_index), dtype=tf.int32))
        num_correct_triggered_fb = tf.reduce_sum(correct_triggered_fb)
        num_correct_triggered_h = tf.reduce_sum(correct_triggered_h)
        num_correct_triggered = tf.reduce_sum(correct_triggered)

        input_dic["num_ground_truth_fb"] = num_gt_fb
        input_dic["num_triggered_fb"] = num_triggered_fb
        input_dic["num_correct_triggered_fb"] = num_correct_triggered_fb
        input_dic["num_ground_truth_h"] = num_gt_h
        input_dic["num_triggered_h"] = num_triggered_h
        input_dic["num_correct_triggered_h"] = num_correct_triggered_h
        input_dic["num_ground_truth"] = num_gt
        input_dic["num_triggered"] = num_triggered
        input_dic["num_correct_triggered"] = num_correct_triggered
        input_dic["batch_precision_triggered"] = num_correct_triggered / num_triggered
        input_dic["batch_recall_triggered"] = num_correct_triggered / num_gt
        #input_dic["sent_length"] = tf.shape(input_dic["raw_sents"])[1]
        #input_dic["fbp"] = input_dic["adjust_p2"]

        return input_dic

    def _compute_debug_info(self, input_dic):
        #compute header and first bullet
        batch_size = tf.shape(input_dic["raw_sents"])[0]
        batch_index = tf.expand_dims(tf.range(batch_size, delta=1, dtype=tf.int32), axis=1)
        h_index = tf.concat([batch_index, tf.cast(tf.expand_dims(input_dic["adjust_p1"], axis=1), tf.int32)], axis=-1)
        fb_index = tf.concat([batch_index, tf.cast(tf.expand_dims(input_dic["adjust_p2"], axis=1), tf.int32)], axis=-1)

        #add one more slot as no answer
        raw_sents_expanded = tf.concat([input_dic["raw_sents"],
                                        tf.expand_dims(tf.tile(tf.constant(["no answer"], dtype=tf.string),
                                                               multiples=[batch_size]),axis=1)],axis=-1)

        headers_output = tf.gather_nd(params=raw_sents_expanded, indices=h_index)
        fb_output = tf.gather_nd(params=raw_sents_expanded, indices=fb_index)
        true_headers = tf.gather_nd(params=raw_sents_expanded, indices=input_dic["header_index"])
        true_fb = tf.gather_nd(params=raw_sents_expanded, indices=input_dic["fb_index"])

        header_compare = tf.concat([tf.expand_dims(headers_output, axis=1), tf.expand_dims(true_headers, axis=1)], axis=-1)
        fb_compare = tf.concat([tf.expand_dims(fb_output, axis=1), tf.expand_dims(true_fb, axis=1)], axis=-1)

        # outputs_text = tf.concat([tf.expand_dims(headers_output, axis=1),
        #                      tf.expand_dims(fb_output, axis=1)], axis=-1)
        outputs_index = tf.concat([tf.expand_dims(input_dic["adjust_p1"], axis=1),
                             tf.expand_dims(input_dic["adjust_p2"], axis=1)], axis=-1)

        input_dic["debug_header_compare"] = header_compare
        input_dic["debug_fb_compare"] = fb_compare
        # input_dic["debug_output_header_fb"] = outputs_text
        input_dic["debug_output_header_fb_index"] = outputs_index
        # input_dic["debug_adjust_p1"] = input_dic["adjust_p1"]
        # input_dic["debug_adjust_p2"] = input_dic["adjust_p2"]




        return input_dic



    def _compute_loss(self, input_dic):
        # input_dic["loss_two_indices"]

        header_scores = tf.concat([input_dic["header_scores"], input_dic["header_no_score"]], axis=-1)
        fb_scores = tf.concat([input_dic["fb_scores"], input_dic["fb_no_score"]], axis=-1)
        h_probs = tf.nn.softmax(header_scores, axis=-1)
        fb_probs = tf.nn.softmax(fb_scores, axis=-1)
        input_dic["combined_h_prob"] = h_probs
        input_dic["combined_fb_prob"] = fb_probs

        #debug([header_scores, input_dic["header_index"], fb_scores, input_dic["fb_index"]], self.list_loader)
        prob_header = tf.gather_nd(h_probs, indices=input_dic["header_index"])
        prob_fb = tf.gather_nd(fb_probs, indices=input_dic["fb_index"])
        loss = tf.reduce_sum(-tf.log(tf.concat([prob_header, prob_fb], axis=-1)))
        input_dic["debug_label_probs"] = tf.concat([prob_header, prob_fb], axis=-1)
        input_dic["debug_header_scores"] = header_scores
        input_dic["debug_fb_scores"] = fb_scores
        input_dic["gather_header_prob"] = prob_header
        input_dic["gather_fb_prob"] = prob_fb

        #debug([loss, input_dic["header_index"], input_dic["fb_index"], h_probs, fb_probs, prob_header, prob_fb], self.list_loader)
                                        # tf.gather_nd(input_dic["a2"], indices=input_dic["fb_index"])
        # num_sents = tf.shape(input_dic["chunk_encoding"])[1]
        # header_index = tf.one_hot(input_dic["header_index"], depth=num_sents, dtype=tf.int32)
        # fb_index = tf.one_hot(input_dic["fb_index"], depth=num_sents, dtype=tf.int32)

        # loss = tf.losses.softmax_cross_entropy(onehot_labels=header_index, logits=input_dic["logits_header"])

        #debug([loss, -tf.log(prob_header), -tf.log(prob_fb), input_dic["fb_index"], prob_fb], self.list_loader)
        # debug([loss], self.list_loader)

        input_dic["loss_with_no_answer"] = loss
        input_dic = self._combine_loss(input_dic)

        # debug([input_dic["loss"]], self.list_loader)

        return input_dic


    def _combine_loss(self, input_dic):

        loss = None
        for i in input_dic:

            if "loss" in i:
                print("loss %s is found" % i)
                if loss is not None:
                    loss += input_dic[i]
                else:
                    loss = input_dic[i]

        input_dic["loss"] = loss

        return input_dic





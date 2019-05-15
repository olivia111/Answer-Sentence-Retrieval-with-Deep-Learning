import tensorflow as tf


#this method is tightly coupled with existing models
class MRCRunOpBuilderV2():


    def __init__(self, num_decoding_steps):

        self.num_decoding_steps = num_decoding_steps

        # self.common_train_ops_names = ["adjust_header_scores",
        #                           "debug_label_probs",
        #                           "adjust_fb_scores",
        #                           "loss",
        #                           "header_em",
        #                           "fb_em",
        #                           "both_em",
        #                           "debug_header_compare",
        #                           "debug_fb_compare",
        #                           "debug_output_header_fb_index",
        #                           "num_ground_truth_fb",
        #                           "num_triggered_fb",
        #                           "num_correct_triggered_fb",
        #                           "num_ground_truth_h",
        #                           "num_triggered_h",
        #                           "num_correct_triggered_h",
        #                           "num_ground_truth",
        #                           "num_triggered",
        #                           "num_correct_triggered"]


        self.common_evaluation_ops_names = ["loss",
                                           "adjust_header_scores",
                                           "adjust_fb_scores",
                                           "adjust_prob_header",
                                           "adjust_prob_fb",
                                           "header_index",
                                           "header_length",
                                           "fb_index",
                                           "fb_length"]


        if num_decoding_steps > 2:
            self.common_evaluation_ops_names.extend(["adjust_sb_scores", "adjust_prob_sb", "sb_index", "sb_length"])

        if num_decoding_steps > 3:
            self.common_evaluation_ops_names.extend(["adjust_tb_scores", "adjust_prob_tb", "tb_index", "tb_length"])


    # def get_train_summary_ops(self, train_ops, computable_summaries, train_input_dic):
    #
    #     # return {"train_ops": train_ops,
    #     #          "computable_summaries": computable_summaries,
    #     #          "adjust_header_scores": train_input_dic["adjust_header_scores"],
    #     #          "debug_label_probs": train_input_dic["debug_label_probs"],
    #     #          "adjust_fb_scores": train_input_dic["adjust_fb_scores"],
    #     #          "loss": train_input_dic["loss"],
    #     #          "h_precision": train_input_dic["batch_precision_header"],
    #     #          "fb_precision": train_input_dic["batch_precision_fb"],
    #     #          "both_precision": train_input_dic["batch_precision_both"],
    #     #          "debug_header_compare": train_input_dic["debug_header_compare"],
    #     #          "debug_fb_compare": train_input_dic["debug_fb_compare"],
    #     #          "debug_output_header_fb_index": train_input_dic["debug_output_header_fb_index"],
    #     #          "num_ground_truth_fb": train_input_dic["num_ground_truth_fb"],
    #     #          "num_triggered_fb": train_input_dic["num_triggered_fb"],
    #     #          "num_correct_triggered_fb": train_input_dic["num_correct_triggered_fb"],
    #     #          "num_ground_truth_h": train_input_dic["num_ground_truth_h"],
    #     #          "num_triggered_h": train_input_dic["num_triggered_h"],
    #     #          "num_correct_triggered_h": train_input_dic["num_correct_triggered_h"],
    #     #          "num_ground_truth": train_input_dic["num_ground_truth"],
    #     #          "num_triggered": train_input_dic["num_triggered"],
    #     #          "num_correct_triggered": train_input_dic["num_correct_triggered"],
    #     #          }
    #     r = {k: train_input_dic[k] for k in self.common_evaluation_ops_names}
    #     r["train_ops"] = train_ops
    #     r["computable_summaries"] = computable_summaries
    #
    #     return r

    # def get_train_print_ops(self, train_ops, train_input_dic):
    #
    #     # return {"train_ops": train_ops,
    #     #          "loss": train_input_dic["loss"],
    #     #          "adjust_header_scores": train_input_dic["adjust_header_scores"],
    #     #          "debug_label_probs": train_input_dic["debug_label_probs"],
    #     #          "adjust_fb_scores": train_input_dic["adjust_fb_scores"],
    #     #          "h_precision": train_input_dic["batch_precision_header"],
    #     #          "fb_precision": train_input_dic["batch_precision_fb"],
    #     #          "both_precision": train_input_dic["batch_precision_both"],
    #     #          "debug_header_compare": train_input_dic["debug_header_compare"],
    #     #          "debug_fb_compare": train_input_dic["debug_fb_compare"],
    #     #          "debug_output_header_fb_index": train_input_dic["debug_output_header_fb_index"],
    #     #          "num_ground_truth_fb": train_input_dic["num_ground_truth_fb"],
    #     #          "num_triggered_fb": train_input_dic["num_triggered_fb"],
    #     #          "num_correct_triggered_fb": train_input_dic["num_correct_triggered_fb"],
    #     #          "num_ground_truth_h": train_input_dic["num_ground_truth_h"],
    #     #          "num_triggered_h": train_input_dic["num_triggered_h"],
    #     #          "num_correct_triggered_h": train_input_dic["num_correct_triggered_h"],
    #     #          "num_ground_truth": train_input_dic["num_ground_truth"],
    #     #          "num_triggered": train_input_dic["num_triggered"],
    #     #          "num_correct_triggered": train_input_dic["num_correct_triggered"],
    #     #          }
    #
    #     r = {k: train_input_dic[k] for k in self.common_evaluation_ops_names}
    #     r["train_ops"] = train_ops
    #
    #     return r

    def get_train_debug_ops(self, train_ops, train_input_dic):

        # gradients_not_none = [i for i in gradients if i is not None]
        # gradients_names = [i.name for i in gradients if i is not None]
        print("debugging mode need to be revised")
        run_ops = {k:v for k,v in train_input_dic.items()}
        # run_ops["gradients"] = gradients_not_none
        run_ops["train_ops"] = train_ops

        return run_ops


    def get_test_pass_through_ops(self, test_input_dic):

        r = {k: test_input_dic[k] for k in self.common_evaluation_ops_names}
        r["raw_query"] = test_input_dic["raw_query"]
        r["raw_sents"] = test_input_dic["raw_sents"]

        return r

    def get_test_run_ops(self, test_input_dic):
        r = {k: test_input_dic[k] for k in self.common_evaluation_ops_names}
        r["raw_query"] = test_input_dic["raw_query"]
        r["raw_sents"] = test_input_dic["raw_sents"]

        return r

    def get_train_print_ops_multiple_batches(self, train_input_dics, train_ops):

        merged_run_ops = {}
        for i in range(len(train_input_dics)):
            run_ops = {k: train_input_dics[i][k] for k in self.common_evaluation_ops_names}
            merged_run_ops[i] = run_ops

        merged_run_ops["loss"] = self.average_dict_tensors("loss", train_input_dics)
        merged_run_ops["train_ops"] = train_ops
        return merged_run_ops, merged_run_ops["loss"]

    def get_train_summary_ops_multiple_batches(self, train_ops, computable_summaries, train_input_dics):

        merged_run_ops, loss = self.get_train_print_ops_multiple_batches(train_input_dics, train_ops)
        merged_run_ops["computable_summaries"] = computable_summaries

        return merged_run_ops, loss



    # def get_merge_train_input_dic(self, train_input_dics, train_ops):



    #produce a merged_test_input_dic
    #by only combine useful ops
    # def get_merge_train_input_dic(self, train_input_dics, train_ops):
    #     merged_train_input_dic = {}
    #     merged_train_input_dic["adjust_header_scores"] = self.build_dict_tensors_as_list("adjust_header_scores", train_input_dics)
    #     merged_train_input_dic["adjust_header_probs"] = self.build_dict_tensors_as_list("adjust_header_probs", train_input_dics)
    #     merged_train_input_dic["header_index"] = self.build_dict_tensors_as_list("header_index", train_input_dics)
    #     merged_train_input_dic["header_length"] = self.build_dict_tensors_as_list("header_length", train_input_dics)
    #     merged_train_input_dic["adjust_fb_scores"] = self.build_dict_tensors_as_list("adjust_fb_scores", train_input_dics)
    #     merged_train_input_dic["adjust_fb_scores"] = self.build_dict_tensors_as_list("adjust_fb_scores",
    #                                                                                  train_input_dics)
    #     merged_train_input_dic["adjust_fb_scores"] = self.build_dict_tensors_as_list("adjust_fb_scores",
    #                                                                                  train_input_dics)
    #     merged_train_input_dic["adjust_fb_probs"] = self.build_dict_tensors_as_list("adjust_fb_probs",
    #                                                                                  train_input_dics)
    #     merged_train_input_dic["fb_index"] = self.build_dict_tensors_as_list("fb_index", train_input_dics)
    #     merged_train_input_dic["fb_length"] = self.build_dict_tensors_as_list("fb_length", train_input_dics)
    #
    #     if self.num_decoding_steps > 2:
    #         merged_train_input_dic["adjust_sb_scores"] = self.build_dict_tensors_as_list("adjust_sb_scores",
    #                                                                                          train_input_dics)
    #         merged_train_input_dic["adjust_sb_probs"] = self.build_dict_tensors_as_list("adjust_sb_probs",
    #                                                                                         train_input_dics)
    #         merged_train_input_dic["sb_index"] = self.build_dict_tensors_as_list("sb_index", train_input_dics)
    #         merged_train_input_dic["sb_length"] = self.build_dict_tensors_as_list("sb_length", train_input_dics)
    #
    #
    #     if self.num_decoding_steps > 3:
    #         merged_train_input_dic["adjust_tb_scores"] = self.build_dict_tensors_as_list("adjust_tb_scores",
    #                                                                                          train_input_dics)
    #         merged_train_input_dic["adjust_tb_probs"] = self.build_dict_tensors_as_list("adjust_tb_probs",
    #                                                                                         train_input_dics)
    #         merged_train_input_dic["tb_index"] = self.build_dict_tensors_as_list("tb_index", train_input_dics)
    #         merged_train_input_dic["tb_length"] = self.build_dict_tensors_as_list("tb_length", train_input_dics)
    #
    #     merged_train_input_dic["train_ops"] = train_ops
    #     merged_train_input_dic["loss"] = self.average_dict_tensors("loss", train_input_dics)
    #     #train ops and summary??
    #
    #     return merged_train_input_dic


    def average_dict_tensors(self, keyword, input_dics):

        r = self.add_dict_tensors(keyword, input_dics) / float(len(input_dics))
        return r

    def concat_dict_tensors(self, keyword, input_dics):

        raise NotImplementedError("concat will bring erros")

        # ts = [i[keyword] for i in input_dics]
        #
        # return tf.concat(ts, axis=0)

    def build_dict_tensors_as_list(self, keyword, input_dics):

        return [i[keyword] for i in input_dics]

    def add_dict_tensors(self, keyword, input_dics):

        assert len(input_dics) > 0

        r = tf.zeros_like(input_dics[0][keyword])
        for i in input_dics:
            r = tf.add(r, i[keyword])

        return r

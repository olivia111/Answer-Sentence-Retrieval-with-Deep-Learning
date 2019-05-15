import tensorflow as tf
from MRC_list_multi_option_loader import MRCListMultiOptionLoader
from MRC_multi_option_model import MRCMultiOptionModel
import os
import math
from learn_rate import LearningRateBuilder
from run_ops_builder_v2 import MRCRunOpBuilderV2
from MRCSummary import MRCSummary
from debug_instances import MRCDebugInstances
from evaluation import EvaluatorMultiOption
import numpy as np

'''
save
checkpoint_dir


train:
continue_train
'''


class MRCListTrainerV2:

    def __init__(self, config, name="MRCListTrainer"):
        self.name = name
        self.config = config
        self.run_ops_builder = MRCRunOpBuilderV2(self.config.model.pointer.num_steps)

    def __call__(self):

        if self.config.global_setting.train_test == "train" or self.config.global_setting.train_test == "train_test":
            self.training()
        elif self.config.global_setting.train_test == "test":
            self.testing()
        elif self.config.global_setting.train_test == "debug":
            mrcdebug = MRCDebugInstances(self.config)
            mrcdebug.start_to_debug()

    def debug_an_instance_complete_detailed(self, run_results, fout):

        self.no_answer_index = -10000

        tags = ["header", "fb", "sb", "tb"]
        probs, scores, label_indices, label_length = [], [], [], []
        raw_queries = run_results["raw_query"].tolist()
        r_sents = run_results["raw_sents"].tolist()
        urls = run_results["raw_url"].tolist()

        for i in range(self.config.model.pointer.num_steps):
            probs.append(run_results["adjust_prob_%s" % tags[i]].tolist())
            scores.append(run_results["adjust_%s_scores" % tags[i]].tolist())
            label_indices.append(run_results["%s_index" % tags[i]].tolist())
            label_length.append(run_results["%s_length" % tags[i]].tolist())

        # print(probs)
        # print(raw_queries)

        last_sent_id = len(probs[0][0]) - 1
        label_indices = [[[k if k != self.no_answer_index else last_sent_id for k in j] for j in i] for i in
                         label_indices]
        # print(label_indices)

        code = 'utf-8-sig'
        for i in range(len(raw_queries)):
            # print(" len of raw sents ", len(raw_sents[i]))
            # print(raw_queries[i])
            # print(raw_sents[i])
            # print(header_compare[i])
            # print(fb_compare[i])
            # print(indices[i])
            # exit(1)
            query = raw_queries[i].decode(code)
            decode_sents = [j.decode(code) for j in r_sents[i]]

            url = urls[i].decode(code)

            sents_to_print = ' '.join([s for s in decode_sents if s != ""])
            # sents = ' '.join([j.decode(code) for j in raw_sents[i] if j != ""])
            sents_to_print = sents_to_print.strip(' ')
            # raw_sents[i].append("no answer")
            decode_sents.append("no answer")

            output_list = [query, url, sents_to_print]
            for j in range(self.config.model.pointer.num_steps):
                probs_with_index = list(enumerate(probs[j][i]))
                no_answer_prob = probs_with_index[-1][1]
                no_answer_score = scores[j][i][-1]
                probs_with_index = sorted(probs_with_index, key=lambda v: v[1], reverse=True)
                item_scores = scores[j][i]

                top1_index, top1_prob, top1, top1_score = probs_with_index[0][0], probs_with_index[0][1], \
                                                          decode_sents[probs_with_index[0][0]], \
                                                          item_scores[probs_with_index[0][0]]
                top2_index, top2_prob, top2, top2_score = probs_with_index[1][0], probs_with_index[1][1], \
                                                          decode_sents[probs_with_index[1][0]], \
                                                          item_scores[probs_with_index[1][0]]
                top3_index, top3_prob, top3, top3_score = probs_with_index[2][0], probs_with_index[2][1], \
                                                          decode_sents[probs_with_index[2][0]], \
                                                          item_scores[probs_with_index[2][0]]

                top1_index = top1_index if top1_index != len(probs_with_index) - 1 else -1
                top2_index = top2_index if top2_index != len(probs_with_index) - 1 else -1
                top3_index = top3_index if top3_index != len(probs_with_index) - 1 else -1

                true_index = label_indices[j][i][0: label_length[j][i]]
                true_prob = [probs[j][i][k] for k in true_index]
                true_score = [scores[j][i][k] for k in true_index]
                # true_h_index = header_ground_truth[i][1]
                # true_fb_index = fb_ground_truth[i][1]
                # true_h_prob = header_probs[i][true_h_index]
                # true_fb_prob = fb_probs[i][true_fb_index]
                # true_h_score = header_scores[i][true_h_index]
                # true_fb_score = fb_scores[i][true_fb_index]

                output_elems = [" ".join(list(map(str, true_index))), " ".join(list(map(str, true_prob))),
                                " ".join(list(map(str, true_score))), no_answer_prob, no_answer_score,
                                top1_index, top1_prob, top1, top1_score, top2_index, top2_prob, top2, top2_score,
                                top3_index, top3_prob, top3, top3_score,
                                str(top1_index in true_index), str(top2_index in true_index),
                                str(top3_index in true_index)]
                output_list.extend(output_elems)

            line = '\t'.join(list(map(str, output_list))) + '\n'
            # line = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%f\t%f\n' %(query, sents, headers, fb, index, header_r, fb_r,
            #                                                 run_results["best_header_prob"][i], run_results["best_fb_prob"][i])
            fout.write(line)
            # exit(1)

    def debug_instances(self, num_instance, run_results):
        header_com = run_results["debug_header_compare"]
        fb_com = run_results["debug_fb_compare"]
        indices = run_results["debug_output_header_fb_index"]

        print("header ===================================")
        print(header_com)
        print("first bullet ================================")
        print(fb_com)
        # print(indices)

    def training(self):

        # np.set_printoptions(threshold=np.nan)

        train_data_loader = MRCListMultiOptionLoader(self.config.train_data, name="train")

        # initialize training process
        if self.config.train.multi_step.enable:
            train_losses, train_input_dics = self.build_train_graph_multibatches(train_data_loader,
                                                                                 self.config.train.multi_step.num_batches)
            learn_rate, train_ops, gradients = self.prepare_train_decay_multibatches(train_losses)
            # train_input_dic = self.run_ops_builder.get_merge_train_input_dic(train_input_dics, train_ops)
            # train_loss = train_input_dic["loss"]
        else:
            # train_loss, train_input_dic = self.build_train_graph(train_data_loader)
            # learn_rate, train_ops, gradients = self.prepare_train_naive_v2(train_loss)
            raise NotImplementedError("only multiple batches mode is enabled")

        # initialize testing while doing training
        test_data_loader = None
        test_input_dic = None
        if self.config.global_setting.train_test == "train_test":
            test_data_loader = MRCListMultiOptionLoader(self.config.test_data, name="test")
            test_loss, test_input_dic = self.build_test_graph(test_data_loader)
        else:
            print("======================no testing=====================")

        self._build_saver()

        eval_module = EvaluatorMultiOption(self.config.model.pointer.num_steps)
        # print(gradients_names)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.tables_initializer())

        # terminal_debug_info = {"loss": self.loss,
        #                        "learn_rate": self.learn_rate,
        #                        "p1": train_input_dic["p1"],
        #                        "p2": train_input_dic["p2"]}

        # build summary for training
        run_train, merged_loss = self.run_ops_builder.get_train_print_ops_multiple_batches(train_input_dics, train_ops)
        train_summary = MRCSummary(None, None, merged_loss)
        computable_summaries = train_summary.get_merged_computable_summaries()

        # computable_summaries = tf.summary.merge_all(scope="computable_summaries")
        # debug_ops = self.run_ops_builder.get_train_debug_ops(train_ops, train_input_dic)

        run_ops_summarize, _ = self.run_ops_builder.get_train_summary_ops_multiple_batches(train_ops,
                                                                                           computable_summaries,
                                                                                           train_input_dics)

        print("start to train")
        debug_mode = self.config.train.debug_mode
        debug_input_data = False
        debug_output = open(self.config.train.debug_output, 'w', encoding='utf-8-sig')
        # sess_config = tf.ConfigProto()
        # sess_config.gpu_options.allow_growth = True
        with tf.Session() as sess:

            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            train_data_loader.start(sess)
            sess.run(init_op)
            # test_data_loader.start(sess)

            train_writer = tf.summary.FileWriter(self.config.train.log_dir, sess.graph)
            train_summary.start(train_writer)

            # print(sess.run(train_input_dic["header_index"]))
            # exit(1)

            if self.config.train.continue_train:
                print("restore check point")
                self.saver.restore(sess, self.train_checkpoint_load_path)

            it = 0

            while True:
                # print(it)
                it += 1

                if self.config.global_setting.train_test == "train_test":
                    if (it % self.config.train.dev_steps) == 1:
                        # pass through the test data once
                        self.pass_through_test_data_once(sess, test_data_loader,
                                                         test_input_dic,
                                                         self.config.train.dev_num_iter,
                                                         train_summary,
                                                         it)

                if debug_input_data:
                    print("iteration %d" % it)
                    run_results = sess.run(train_input_dics[0]["chunk_encoding"])
                    print(np.shape(run_results))
                    continue


                elif debug_mode:
                    print("currently no debug mode available")
                    exit(1)
                else:

                    if (it % self.config.train.summary_steps) == 0:
                        # run summary op
                        run_results = sess.run(run_ops_summarize)
                    else:
                        run_results = sess.run(run_train)

                    for n_b in range(self.config.train.multi_step.num_batches):
                        eval_module.add_one_result(run_results[n_b])

                    if (it % self.config.train.summary_steps) == 0:
                        # print summary
                        train_writer.add_summary(run_results["computable_summaries"], it)
                        accpr_results = eval_module.get_results_multi_iterations(
                            self.config.train.multi_step.num_batches)
                        train_summary.add_all_AccPR(accpr_results, it, "train")

                    if it % 10 == 0:
                        # debug
                        loss_val = run_results["loss"]
                        print("PROGRESS it %d train loss %f " % (it, loss_val))
                        eval_module.print_results_multi_iterations(self.config.train.multi_step.num_batches,
                                                                   "current_iter")
                        eval_module.print_results_multi_iterations(self.config.train.multi_step.num_batches * 10,
                                                                   "last 10")

                        if self.config.train.print_instances:
                            raise NotImplementedError("debug disabled")
                        # self.debug_instances(1, run_results)

                if "loss" in run_results and run_results["loss"] == float("inf"):
                    print("it %d train loss %f " % (it, run_results["loss"]))
                    # print(run_results["adjust_header_scores"])
                    # print(run_results["adjust_header_scores"])
                    # print(run_results["debug_label_probs"])

                    break

                if "loss" in run_results and math.isnan(run_results["loss"]):
                    print("loss is nan")
                    break

                # sess.run(train_ops)

                # if it % 10 == 0:
                #     print("it %d test loss %f " %(it, sess.run(test_loss)))

                if it % self.config.train.num_steps_to_save == 0:
                    checkoutpath = os.path.join(self.train_checkpoint_save_dir, "model.ckpt")
                    print("checkout path ", checkoutpath)
                    self.saver.save(sess, checkoutpath,
                                    global_step=it)
        debug_output.close()

    def pass_through_test_data_once(self, sess, test_data_loader, test_input_dic, num_dev_iters, train_summary,
                                    train_it):

        test_data_loader.start(sess)

        test_one_pass_eval_module = EvaluatorMultiOption(self.config.model.pointer.num_steps)

        print("==============================testing==============================")

        # print("restore check point")
        # self.saver.restore(sess, self.test_checkpoing_load_path)

        # evaluator.testing_using_checkpoint(sess=sess,
        #                                    dataloader=test_data_loader,
        #                                    input_dic=test_input_dic)

        test_one_file_run_ops = self.run_ops_builder.get_test_pass_through_ops(test_input_dic)

        it = 0
        loss_mean = 0.0
        try:
            for i in range(num_dev_iters):
                run_results = sess.run(test_one_file_run_ops)
                loss_mean += run_results["loss"]
                test_one_pass_eval_module.add_one_result(run_results)
                it += 1

        except tf.errors.OutOfRangeError:
            print("reach the end of evaluation set")

        print("number of iteration eval %d", it)
        assert it > 0

        test_one_pass_eval_module.print_results_multi_iterations(-1)
        accpr_results = test_one_pass_eval_module.get_results_multi_iterations(-1)
        loss_mean = loss_mean / float(num_dev_iters)
        print("loss is ", loss_mean)

        train_summary.add_all_AccPR(accpr_results, train_it, "test_onepass")
        train_summary.add_a_pseudo_summary_value("test_onepass_loss", loss_mean, train_it)

        print("==============================done==============================")

    def testing(self):
        test_data_loader = MRCListMultiOptionLoader(self.config.test_data, name="test")
        test_loss, test_input_dic = self.build_test_graph(test_data_loader)

        self._build_saver()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.tables_initializer())

        # evaluator = MRCEvaluator(self.config.test, name="test_exp")
        num_iters = self.config.test.num_iteration
        eval_module = EvaluatorMultiOption(self.config.model.pointer.num_steps)

        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            test_data_loader.start(sess)
            sess.run(init_op)

            print("restore check point")

            self.saver.restore(sess, self.test_checkpoint_load_path)
            test_run_ops = self.run_ops_builder.get_test_run_ops(test_input_dic)
            it = 0
            test_example_file = open(self.config.test.test_results, 'w', encoding='utf-8-sig')
            ##query_token clean_context pred_header true_header pred_fb true_fb header_right? fb_right?
            if self.config.test.mode == "simple":
                test_example_file.write(
                    "query_token\tjoin_context\t[pred_header,true_header]\t[pred_fb,true_fb]\tpred_indices\theader_right?\tfb_right?\theader_score\tfb_score\n")
            elif self.config.test.mode == "detailed":

                header = ["query_token", "url_clean", "join_context"]
                tags = ["header", "fb", "sb", "tb"]
                for k in range(self.config.model.pointer.num_steps):
                    elems = ["%s_index" % tags[k], "%s_prob" % tags[k], "%s_score" % tags[k],
                             "no_answer_%s_prob" % tags[k],
                             "no_answer_%s_score" % tags[k], "top1_%s_index" % tags[k], "top1_%s_prob" % tags[k],
                             "top1_%s" % tags[k],
                             "top1_%s_score" % tags[k], "top2_%s_index" % tags[k], "top2_%s_prob" % tags[k],
                             "top2_%s" % tags[k],
                             "top2_%s_score" % tags[k], "top3_%s_index" % tags[k], "top3_%s_prob" % tags[k],
                             "top3_%s" % tags[k],
                             "top3_%s_score" % tags[k], "top1_%s_right" % tags[k], "top2_%s_right" % tags[k],
                             "top3_%s_right" % tags[k]]
                    header.extend(elems)

                test_example_file.write('\t'.join(header) + '\n')
            try:
                for i in range(num_iters):
                    it += 1
                    run_results = sess.run(test_run_ops)
                    eval_module.add_one_result(run_results)

                    if self.config.test.mode == "simple":
                        raise NotImplementedError("simple test results not implemented")
                        # self.debug_an_instance_complete(run_results, test_example_file)
                    elif self.config.test.mode == "detailed":
                        self.debug_an_instance_complete_detailed(run_results, test_example_file)

                    if it % 10 == 0:
                        print("it %d train loss %f " % (it, run_results["loss"]))
                        eval_module.print_results_multi_iterations(self.config.train.multi_step.num_batches)


            except tf.errors.OutOfRangeError:
                print("reach the end of evaluation set")

            test_example_file.close()

            print("number of iteration eval %d", it)
            assert it > 0
            print("final metric")
            eval_module.print_results_multi_iterations(-1)

    def prepare_training_decay_rate(self, train_loss, global_step):

        with tf.variable_scope("decay_training"):
            learn_rate = LearningRateBuilder(self.config.train.learn_rate, global_step=global_step)
            optimizer = self._build_optimizer(learn_rate)  # compute the probability of best header and first bullet

            global_step = tf.get_variable(name="global_step", dtype=tf.int32, initializer=tf.constant(0))
            params, gradients = self._build_gradient(train_loss)
            train_ops = optimizer.apply_gradients(zip(gradients, params), global_step, name="apply_gradients")

        return learn_rate, train_ops

    def prepare_train_naive_v2(self, train_loss):
        learn_rate = tf.constant(self.config.train.learn_rate.init_rate)
        optimizer = self._build_optimizer(self.config.train.learn_rate.init_rate)
        params, gradients = self._build_gradient(train_loss)
        train_ops = optimizer.apply_gradients(zip(gradients, params))
        return learn_rate, train_ops, gradients

    def prepare_training_naive(self, train_loss):
        learn_rate = tf.constant(self.config.train.learn_rate.init_rate)
        optimizer = self._build_optimizer(learn_rate)
        train_ops = optimizer.minimize(loss=train_loss)
        return learn_rate, train_ops

    def build_train_graph(self, train_data_loader):
        # word embedding
        train_input_dic = train_data_loader.get_batch()

        # debug([train_input_dic["query2chars"], train_input_dic["sents2chars"]], train_data_loader)
        global_config = {"enable_summary": True}

        train_input_dic = MRCMultiOptionModel(self.config.model, global_config, name="MRCmodel")(train_input_dic,
                                                                                                 train_data_loader)
        train_loss = train_input_dic["loss"]
        return train_loss, train_input_dic

    def _build_gradient(self, train_loss):

        params = tf.trainable_variables()
        # gradients = tf.gradients(train_loss, params)
        gradients = self._clip_gradient(tf.gradients(train_loss, params))

        # print(gradients)
        if self.config.train.summary_enable:
            with tf.name_scope("train_summaries"):
                with tf.name_scope("gradients"):
                    for v in gradients:
                        if v is not None:
                            tf.summary.histogram(v.name.replace(":", "_"), v)

        return params, gradients

    def _build_optimizer(self, learn_rate):

        if self.config.train.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        else:
            raise NotImplementedError

        return optimizer

    def build_test_graph(self, test_data_loader):

        # check some parameters
        self.config.model.assign_a_value_for_all_key_with("dropout_type", "None")
        print("testing has dropout type ", self.config.model.query_encoding.dropout_type)

        global_config = {"enable_summary": False}
        test_input_dic = test_data_loader.get_batch()[0]
        test_input_dic = MRCMultiOptionModel(self.config.model, global_config, name="MRCmodel")(test_input_dic,
                                                                                                test_data_loader)
        test_loss = test_input_dic["loss"]
        return test_loss, test_input_dic

    def _build_saver(self):

        # self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=3)
        # self.check_point_path = tf.train.latest_checkpoint(self.config.train.cbuild_train_graph_multibatchesheckpoint_dir)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
        print(self.config.train.checkpoint_load_path)
        # if os.path.isfile(self.config.train.checkpoint_load_path):
        # self.train_checkpoint_load_path = self.config.train.checkpoint_load_path
        # elif os.path.isdir(self.config.train.checkpoint_load_path):
        # self.train_checkpoint_load_path = tf.train.latest_checkpoint(self.config.train.checkpoint_load_path)
        # else:
        # raise NotImplementedError("not a file or a dir?")
        self.train_checkpoint_load_path = tf.train.latest_checkpoint(self.config.train.checkpoint_load_path)

        self.train_checkpoint_save_dir = self.config.train.checkpoint_save_dir

        if "ckpt" in self.config.test.checkpoint_load_path:
            print("loading ckpt file")
            self.test_checkpoint_load_path = self.config.test.checkpoint_load_path
        else:
            print("loading checkpoing in folder")
            self.test_checkpoint_load_path = tf.train.latest_checkpoint(self.config.test.checkpoint_load_path)

            print(self.test_checkpoint_load_path)

    def _clip_gradient(self, raw_gradients):

        if self.config.train.clip_mode == "global_norm":
            clipped_gradients, norm = tf.clip_by_global_norm(raw_gradients, self.config.train.clip_norm)
        elif self.config.train.clip_mode == "norm":
            clipped_gradients = [tf.clip_by_norm(g, self.config.train.clip_norm) if g is not None else None
                                 for g in raw_gradients]
        elif self.config.train.clip_mode == "None":
            clipped_gradients = raw_gradients
        else:
            raise NotImplementedError("clip gradient")

        return clipped_gradients

    # start here, doing multiple batch train
    def build_train_graph_multibatches(self, train_data_loader, num_batches):
        global_config = {"enable_summary": True}

        losses = []
        train_input_dics = train_data_loader.get_batch(num_batches)
        for i in range(num_batches):
            print("building training graph %d==============================" % i)

            train_input_dics[i] = MRCMultiOptionModel(self.config.model, global_config, name="MRCmodel")(
                train_input_dics[i],
                train_data_loader)
            t_train_loss = train_input_dics[i]["loss"]
            losses.append(t_train_loss)

        return losses, train_input_dics

    def prepare_train_naive_multibatches(self, train_losses):

        learn_rate = tf.constant(self.config.train.learn_rate.init_rate)
        optimizer = self._build_optimizer(self.config.train.learn_rate.init_rate)
        params, gradients = self._build_gradient_multibatches(train_losses)
        train_ops = optimizer.apply_gradients(zip(gradients, params))

        return learn_rate, train_ops, gradients

    def _build_gradient_multibatches(self, train_losses):

        params = tf.trainable_variables()

        # print(params)
        accum_grad = [tf.zeros_like(var) for var in params]

        for train_l in train_losses:
            accum_grad = [tf.add(accum_grad[i], grad) for i, grad in enumerate(tf.gradients(train_l, params))]

        # exit(1)
        # grad_mean = []

        # all_gradients = [tf.gradients(train_l, params) for train_l in train_losses]
        # grad_mean = [0 for _ in params]
        # for train_l in train_losses:
        #     grad = tf.gradients(train_l, params)
        #     print(grad)
        #     for i in range(len(grad)):
        #         grad_mean[i] += grad[i]

        # num_batches_to_div = float(len(train_losses))
        # for i in range(len(grad_mean)):
        #     grad_mean[i] = grad_mean[i] / num_batches_to_div

        # gradients = tf.gradients(train_loss, params)
        gradients = self._clip_gradient(accum_grad)

        # print(gradients)
        # if self.config.train.summary_enable:
        #     with tf.name_scope("train_summaries"):
        #         with tf.name_scope("gradients"):
        #             for v in gradients:
        #                 if v is not None:
        #                     tf.summary.histogram(v.name.replace(":", "_"), v)

        return params, gradients

    def prepare_train_decay_multibatches(self, train_losses):
        with tf.variable_scope("decay_training"):
            #global_step = tf.get_variable(name="global_step", dtype=tf.int32, initializer=tf.constant(0))
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
            learn_rate = LearningRateBuilder(self.config.train.learn_rate, global_step=global_step)
            lr = learn_rate()
            optimizer = self._build_optimizer(lr)
            params, gradients = self._build_gradient(train_losses)
            train_ops = optimizer.apply_gradients(zip(gradients, params), global_step, name="apply_gradients")

        return learn_rate, train_ops, gradients







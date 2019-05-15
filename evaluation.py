import numpy as np
from metric import compute_PR

class EvaluatorMultiOption:

    def __init__(self, num_decoding_steps):
        self.num_decoding_steps = num_decoding_steps

        self.num_correct_triggers = []
        self.num_triggereds = []
        self.num_ground_truths = []
        self.num_exact_matches = []
        # self.precisions = []
        # self.recalls = []
        # self.accs = []



    # def get_aggregrate_results(self, num_iters):
    #     metric_values = {}
    #     return metric_values

    def add_one_result(self, run_results):

        def compute_label_match(labels, pred, label_length):
            labels_list = labels.tolist()
            pred_list = pred.tolist()
            label_length_list = label_length.tolist()
            #both are list
            correct_pred = []
            num_ground_truth = 0
            for batch_index in range(len(labels_list)):
                correct_pred.append(1 if pred_list[batch_index] in labels_list[batch_index][0:label_length_list[batch_index]] else 0)
                if -1 not in labels_list[batch_index][0:label_length_list[batch_index]]:
                    num_ground_truth += 1
            # print(labels_list)
            # print(pred)
            # print(correct_pred)
            # print(num_ground_truth)
            # exit(1)
            return np.array(correct_pred), num_ground_truth


        probs, scores, labels, label_lengths = [], [], [], []


        probs.append(run_results["adjust_prob_header"])
        scores.append(run_results["adjust_header_scores"])
        labels.append(run_results["header_index"])
        label_lengths.append(run_results["header_length"])

        probs.append(run_results["adjust_prob_fb"])
        scores.append(run_results["adjust_fb_scores"])
        labels.append(run_results["fb_index"])
        label_lengths.append(run_results["fb_length"])

        if self.num_decoding_steps > 2:
            probs.append(run_results["adjust_prob_sb"])
            scores.append(run_results["adjust_sb_scores"])
            labels.append(run_results["sb_index"])
            label_lengths.append(run_results["sb_length"])


        if self.num_decoding_steps > 3:
            probs.append(run_results["adjust_prob_tb"])
            scores.append(run_results["adjust_tb_scores"])
            labels.append(run_results["tb_index"])
            label_lengths.append(run_results["tb_length"])


        # print(probs)
        # print(scores)
        # print(labels)
        # print(label_lengths)

        num_correct_triggers = []
        num_triggereds = []
        num_ground_truths = []
        num_exact_matches = []
        # ps, rs, accs = [], [], []

        self.batch_size = np.shape(scores[0])[0]

        for i in range(self.num_decoding_steps):
            num_sents = np.shape(scores[i])[-1] - 1
            preds = np.argmax(scores[i], axis=-1)

            # print(preds)
            # print(scores[i])
            preds[preds == num_sents] = -1

            # print(labels)

            # print(preds == num_sents)
            # print(num_sents)
            # print(preds)
            # print(preds_replaced)

            #at this point, label  use -1, pred uses num sents. They will not match
            correct_pred, num_ground_truth = compute_label_match(labels[i], preds, label_lengths[i])
            # print(correct_pred)
            num_correct_trigger = np.count_nonzero(np.logical_and(correct_pred == 1, preds != -1))
            # num_correct_trigger = np.sum(correct_pred)
            num_triggered = np.count_nonzero(preds != -1)
            # num_ground_truth = np.count_nonzero(labels[i] != -1)
            num_exact_match = np.sum(correct_pred)

            # print(num_correct_trigger)
            # print(num_triggered)
            # print(num_exact_match)
            num_correct_triggers.append(num_correct_trigger)
            num_triggereds.append(num_triggered)
            num_ground_truths.append(num_ground_truth)
            num_exact_matches.append(num_exact_match)

            if num_correct_trigger > num_ground_truth:
                print("number of correct trigger greater than num ground_truth")

            # p, r = compute_PR(num_triggered, num_correct_trigger, num_ground_truth)
            # acc = float(num_exact_match)/float(self.batch_size)
            # ps.append(p)
            # rs.append(r)
            # accs.append(acc)

            # if num_exact_match == self.batch_size and p < 1.0:
            #     print(num_correct_trigger)
            #     print(correct_pred)
            #     print(num_triggered)
            #     print(num_exact_match)
            #     print(num_ground_truth)
            #     exit(1)

        # self.precisions.append(ps)
        # self.recalls.append(rs)
        # self.accs.append(accs)

        # print(self.precisions)
        # print(self.recalls)
        # print(self.accs)
        # exit(1)
        self.num_correct_triggers.append(num_correct_triggers)
        self.num_triggereds.append(num_triggereds)
        self.num_ground_truths.append(num_ground_truths)
        self.num_exact_matches.append(num_exact_matches)



    def print_results_current_iteration(self):
        for i in range(self.num_decoding_steps):
            p, r = compute_PR(self.num_triggereds[-1][i], self.num_correct_triggers[-1][i], self.num_ground_truths[-1][i])
            print("current %d  p:%f  r:%f  acc:%f" %(i, p, r, self.num_exact_matches[-1][i]/float(self.batch_size)))


    def print_results_multi_iterations(self, num_iters, string_prefix=""):

        results = self.get_results_multi_iterations(num_iters)

        for i in range(self.num_decoding_steps):
            print("%s %d  %d  p:%f  r:%f  acc:%f" % (string_prefix, num_iters, i, results[i]["p"], results[i]["r"], results[i]["acc"]))

        return results

    def get_results_multi_iterations(self, num_iters):
        #not printing
        if num_iters == -1:
            start_from = 0
            num_iters = len(self.num_correct_triggers)
        elif num_iters > len(self.num_correct_triggers):
            start_from = 0
            num_iters = len(self.num_correct_triggers)
        else:
            start_from = len(self.num_correct_triggers) - num_iters

        results = []
        # print(self.accs)
        # print(num_iters)

        for i in range(self.num_decoding_steps):
            acc_num_triggered = sum([self.num_triggereds[j][i] for j in range(start_from, len(self.num_triggereds))])
            acc_num_correct_trigger = sum([self.num_correct_triggers[j][i] for j in range(start_from, len(self.num_correct_triggers))])
            acc_num_ground_truths = sum([self.num_ground_truths[j][i] for j in range(start_from, len(self.num_ground_truths))])
            acc_num_exact_match = sum([self.num_exact_matches[j][i] for j in range(start_from, len(self.num_exact_matches))])
            p, r = compute_PR(acc_num_triggered, acc_num_correct_trigger, acc_num_ground_truths)
            acc = acc_num_exact_match / float(num_iters * self.batch_size)
            results.append({"acc":acc, "p":p, "r":r})

        return results
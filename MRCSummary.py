import tensorflow as tf


#this class is tightly coupled with MRC trainer
class MRCSummary:




    def __init__(self, train_input_dic, test_input_dic, train_loss):

        self.computable_summary_name = "computable_summaries"

        self.pseudo_summaries = {}
        self.computable_summaries = {}

        #build summary
        self._build_computable_summaries(train_loss)
        self._build_pseudo_summaries()
        self.train_writer = None


    def _build_computable_summaries(self, train_loss):

        with tf.name_scope(self.computable_summary_name):
            self.computable_summaries["loss"] = tf.summary.scalar("loss", train_loss)


    def _build_pseudo_summaries(self):

        pseudo_summaries = {}
        with tf.name_scope("training_pseudo"):

            for t in ["train", "test_onepass"]:
                for s in ["header", "fb", "sb", "tb", "both"]:
                    pseudo_summaries["%s_%s_em"%(t, s)] = tf.Summary()
                    pseudo_summaries["%s_%s_em"%(t, s)].value.add(tag="%s_%s_em"%(t, s), simple_value=None)
                    pseudo_summaries["%s_%s_precision"%(t,s)] = tf.Summary()
                    pseudo_summaries["%s_%s_precision"%(t,s)].value.add(tag="%s_%s_precision"%(t,s), simple_value=None)
                    pseudo_summaries["%s_%s_recall"%(t,s)] = tf.Summary()
                    pseudo_summaries["%s_%s_recall"%(t,s)].value.add(tag="%s_%s_recall"%(t,s), simple_value=None)

            pseudo_summaries["test_onepass_loss"] = tf.Summary()
            pseudo_summaries["test_onepass_loss"].value.add(tag='test_onepass_loss', simple_value=None)

        self.pseudo_summaries = pseudo_summaries

    def add_a_pseudo_summary_value(self, name, value, iteration):

        #this method
        if self.train_writer is None:
            raise Exception("summary needs to start")

        self.pseudo_summaries[name].value[0].simple_value = value
        self.train_writer.add_summary(self.pseudo_summaries[name], iteration)

    def add_all_AccPR(self, metric_results, iteration, string_prefix="train"):

        tags = ["header", "fb", "sb", "tb"]
        for i in range(len(metric_results)):

            self.add_a_pseudo_summary_value("%s_%s_em"%(string_prefix, tags[i]), metric_results[i]["acc"], iteration)
            self.add_a_pseudo_summary_value("%s_%s_precision" % (string_prefix, tags[i]), metric_results[i]["p"], iteration)
            self.add_a_pseudo_summary_value("%s_%s_recall" % (string_prefix, tags[i]), metric_results[i]["r"], iteration)



    def start(self, train_writer):
        self.train_writer = train_writer


    def get_pseudo_summaries(self):
        return self.pseudo_summaries

    def get_merged_computable_summaries(self):
        return tf.summary.merge([v for k, v in self.computable_summaries.items()])




# class MRCMultiStepSummary(MRCSummary):
#
#
#     def __init__(self):
#
#         self.computable_summary_name = "computable_summaries"
#
#         _build_multi_step_computable_summaries()
#         self.train_writer = None
#
#
#     def build_multistep_summary(self, train_input_dic, test_input_dic, train_loss):
#         self._build_computable_summaries(train_loss)
#         self._build_pseudo_summaries()
#
#     def _build_multi_step_computable_summaries(self, train_losses):

import tensorflow as tf
from MRC_list_multi_option_loader import MRCListMultiOptionLoader
from MRC_multi_option_model import MRCMultiOptionModel
from fetch_info import FetchInfoFromModel


class MRCDebugInstances:

    def __init__(self, config):
        self.config = config
        self.run_ops = {}
        self.fetch_info = FetchInfoFromModel(mode="debug_instance")

    def _add_common_ops_from_input_dic(self, run_ops, test_input_dic):

        run_ops["raw_query"] = test_input_dic["raw_query"]
        run_ops["raw_sents"] = test_input_dic["raw_sents"]
        run_ops["header_index"] = test_input_dic["header_index"]
        run_ops["fb_index"] = test_input_dic["fb_index"]
        run_ops["loss"] = test_input_dic["loss"]

    def start_to_debug(self):

        print("starting debug mode=============================================")
        print("intent to only run one example====================================")
        test_data_loader, run_ops, test_input_dic = self.build_debug_graph()
        self._add_common_ops_from_input_dic(run_ops, test_input_dic)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.tables_initializer())
        self._build_saver()

        with tf.Session() as sess:
            test_data_loader.start(sess)

            sess.run(init_op)
            print("restore check point")
            self.saver.restore(sess, self.test_checkpoint_load_path)

            run_results = sess.run(run_ops)
            self.print_results(run_results)


    def build_debug_graph(self):

        #check some parameters
        test_data_loader = MRCListMultiOptionLoader(self.config.test_data, name="test")
        self.config.model.assign_a_value_for_all_key_with("dropout_type", "None")
        self.config.test_data.batch_size = 1
        print("testing has dropout type ", self.config.model.query_encoding.dropout_type)

        global_config = {"enable_summary": False}
        test_input_dic = test_data_loader.get_batch()[0]
        test_input_dic = MRCMultiOptionModel(self.config.model,
                                  global_config,
                                  name="MRCmodel",
                                  fetch_info=self.fetch_info)(test_input_dic, test_data_loader)

        run_ops = self.fetch_info.get_run_ops()
        return test_data_loader, run_ops, test_input_dic

    def add_a_debug_op(self, key, tensor_unit):
        assert key not in self.run_ops

        self.run_ops[key] = tensor_unit

    def print_results(self, run_results):

        debug_file = r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\data\debug_print.txt"

        with open(debug_file, 'w', encoding='utf-8-sig') as fout:

            line = '========================================================'
            twoemptyline = "\n\n"

            #first print query and sentences
            #make sure only one batch
            query = run_results["raw_query"][0]
            fout.write("Query  :   %s \n" %query)
            sents = run_results["raw_sents"][0]

            fout.write("Sentences %s\n" %line)
            for i, v in enumerate(sents):
                fout.write("%d        %s\n" %(i, v))

            fout.write("pointer attention\%s\n" %line)
            fout.write("pointer_decoding_1_softmax\n")
            fout.write(str(run_results["pointer_decoding_1_softmax"]) + '\n')
            fout.write(twoemptyline)

            fout.write("pointer_decoding_2_softmax\n")
            fout.write(str(run_results["pointer_decoding_2_softmax"]) + '\n')
            fout.write(twoemptyline)

            fout.write("sents encoding %s\n" %line)
            fout.write("sents_encoding_attn_softmax\n")
            fout.write(str(run_results["sents_encoding_attn_softmax"]) + '\n')
            fout.write(twoemptyline)

            fout.write("sents_encoding_attn_gates\n")
            fout.write(str(run_results["sents_encoding_attn_gate"]) + '\n')
            fout.write(twoemptyline)

            fout.write("max_mean_sents_pooling_argmax%s\n" %line)
            fout.write(str(run_results["max_mean_sents_pooling_argmax"]) + '\n')
            fout.write(twoemptyline)

            fout.write("qc_attention %s\n" %line)
            fout.write("qc_attention_attn_softmax\n")
            fout.write(str(run_results["qc_attention_attn_softmax"]) + '\n')
            fout.write(twoemptyline)

            fout.write("qc_attention_attn_gates\n")
            fout.write(str(run_results["qc_attention_attn_gate"]) + '\n')
            fout.write(twoemptyline)

            fout.write("self_attention%s\n" %line)
            fout.write("self_attention_attn_gates\n")
            fout.write(str(run_results["self_attention_attn_gate"]) + '\n')
            fout.write(twoemptyline)

            fout.write("self_attention_attn_softmax\n")
            fout.write(str(run_results["self_attention_attn_softmax"]) + '\n')
            fout.write(twoemptyline)

    #copy from train_v2
    def _build_saver(self):


        # self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=3)
        # self.check_point_path = tf.train.latest_checkpoint(self.config.train.cbuild_train_graph_multibatchesheckpoint_dir)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
        print(self.config.train.checkpoint_load_path)
        #if os.path.isfile(self.config.train.checkpoint_load_path):
            #self.train_checkpoint_load_path = self.config.train.checkpoint_load_path
        #elif os.path.isdir(self.config.train.checkpoint_load_path):
            #self.train_checkpoint_load_path = tf.train.latest_checkpoint(self.config.train.checkpoint_load_path)
        #else:
            #raise NotImplementedError("not a file or a dir?")
        self.train_checkpoint_load_path = tf.train.latest_checkpoint(self.config.train.checkpoint_load_path)

        self.train_checkpoint_save_dir = self.config.train.checkpoint_save_dir

        #if os.path.isfile(self.config.test.checkpoint_load_path):
            #self.test_checkpoing_load_path = self.config.test.checkpoint_load_path
        #elif os.path.isdir(self.config.test.checkpoint_load_path):
            #self.test_checkpoing_load_path = tf.train.latest_checkpoint(self.config.test.checkpoint_load_path)
        #else:
            #raise NotImplementedError("not a file or a dir?")
        self.test_checkpoint_load_path = tf.train.latest_checkpoint(self.config.test.checkpoint_load_path)



import platform
import sys
import argparse
import os
import sys
from config import Config
from trainer_v2 import MRCListTrainerV2
import tensorflow as tf

def add_root_path():
    cur_file_path = os.path.realpath(__file__)
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(cur_file_path)))
    print(root_path)
    sys.path.append(root_path)

    return root_path

def test_path(args):

    print(args.train_data_dir)
    print(args.test_data_dir)
    print(args.out_model_path)
    #print(args.prev_model_path)
    print(args.log_dir)


def overload_params():

    parser = argparse.ArgumentParser()

    #input data
    #these paths will be replaced by aether
    parser.add_argument('--input-training-data-path', type=str, nargs='?',
                        dest='train_data_dir', default=r"C:\Users\silin\Downloads\MRC_table\data",
                        help="a folder containing training data and model_data")
    parser.add_argument('--input-validation-data-path', type=str, nargs='?', help='path of testing data', dest='test_data_dir')
    parser.add_argument('--output-model-path', type=str, help='path of output model', dest='out_model_path',
                        default=r"C:\Users\silin\Downloads\MRC_table\data\output_checkpoint")
    parser.add_argument('--input-previous-model-path', type=str, help='path of previous model',
                        dest='prev_model_dir', default=r"C:\Users\silin\Downloads\MRC_table\data\checkpoint")
    parser.add_argument('--log-dir', type=str, help='path of log', dest='log_dir',
                        default=r"C:\Users\silin\Downloads\MRC_table\data")
    parser.add_argument("--direct_test_checkpoint_path", type=str, default="None")

    parser.add_argument('--prev_model_name', type=str, default="")
    parser.add_argument('--train_test', type=str, default='train')
    parser.add_argument('--continue_train', type=str, default="False")
    parser.add_argument('--init_learn_rate', type=float, default=0.001)
    parser.add_argument('--train_data_file', type=str,
                        default=r'uhrs_matching_train_filter.tsv')
    parser.add_argument('--test_data_file', type=str,
                        default=r'uhrs_matching_test_filter.tsvt')
    parser.add_argument('--config_file', type=str, default='big_bi_test_no_qattn.yml')
    parser.add_argument('--extra_config_file', type=str, default="None")

    # params
    parser.add_argument('--clip_mode', type=str, default='global_norm')
    parser.add_argument('--clip_norm', type=float, default=5)
    # parser.add_argument('--init_learn_rate', type=float, default=0.001)
    parser.add_argument('--unified_dropout_keep_prob', type=float, default=0.9)
    parser.add_argument('--unified_cell_type', type=str, default='lstm')
    # assign this later
    parser.add_argument('--dropout_keep_prob_verifier', type=float, default=0.5)
    parser.add_argument('--dropout_keep_prob_word', type=float, default=0.8)
    # parser.add_argument('--unified_atten_type', type=str, default='rnet')

    # logging
    parser.add_argument('--summary_steps', type=int, default=-1)

    # change the model
    parser.add_argument('--sent_pooling_type', type=str, default='max_mean')

    # model/ newly added
    parser.add_argument("--loss_weights", type=str, default="1_1_1_1")
    parser.add_argument("--voc_type", type=str, default="45k")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--train_num_batch_group", type=int, default=1)
    parser.add_argument("--train_dev_num_iter", type=int, default=1000)
    parser.add_argument("--train_dev_steps", type=int, default=1000)
    parser.add_argument("--train_num_steps_to_save", type=int, default=5000)
    parser.add_argument('--word_embedding_trainable', type=str, default="False")

    # filter
    parser.add_argument("--filter_max_num_sents_train", type=int, default=81)
    parser.add_argument("--filter_max_num_sents_test", type=int, default=81)
    parser.add_argument("--dropout_type", type=str, default="var")

    # for brainwave quantization
    parser.add_argument("--override_quant_mode", type=str, default="None")

    args = parser.parse_args()

    # process bool variable
    print(args.word_embedding_trainable)
    args.word_embedding_trainable = True if args.word_embedding_trainable == "True" else False
    args.continue_train = True if args.continue_train == "True" else False

    print("printing parameters")
    print("train_data_dir %s" % args.train_data_dir)
    print("prev_model_dir %s" % args.prev_model_dir)
    print("train_data_file %r" % args.train_data_file)
    print("test_data_file %r" % args.test_data_file)
    print("train_test %r" % args.init_learn_rate)
    print("train_test %s" % args.train_test)
    print("voc_type %s" % args.voc_type)
    print("log_dir %s" % args.log_dir)
    print("out_model_path %s" % args.out_model_path)
    print("loss_weights %s" % args.loss_weights)
    print("train_num_batch_group %d" % args.train_num_batch_group)
    print("train_batch_size %d" % args.train_batch_size)
    print("args.train_dev_num_iter %d" % args.train_dev_num_iter)
    print("args.continue_train %s" % str(args.continue_train))
    print("args.override_quant_mode %s" % args.override_quant_mode)
    print("word_embedding_trainable %s" % str(args.word_embedding_trainable))

    root_path = add_root_path()

    # print(args.train_data)
    test_folder = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(os.path.join(test_folder, "config_files"), args.config_file)
    # config_path = os.path.join(test_folder, args.config_file)
    # extra config dir is root_path
    config = Config.create_config(config_path)

    if args.extra_config_file != "None":
        extra_config_path = os.path.join(root_path, args.extra_config_file)
        config.override(extra_config_path)

    # replace paths
    # newly added
    if args.voc_type == "45k":
        voc_data_dic = 'glove.6B.300d.45k.voc.data'
        voc_data_embedding = 'glove.6B.300d.45k.data'
    else:
        voc_data_dic = "glove.6B.300d.200k.voc.data"
        voc_data_embedding = 'glove.6B.300d.200k.data'

    config.train.multi_step.num_batches = args.train_num_batch_group
    config.train_data.batch_size = args.train_batch_size

    if args.train_dev_num_iter >= 0:
        config.train.dev_num_iter = args.train_dev_num_iter
    if args.train_dev_steps >= 0:
        config.train.dev_steps = args.train_dev_steps
        config.train.num_steps_to_save = args.train_dev_steps

    model_data_folder = os.path.join(args.train_data_dir, 'model_data')
    data_folder = args.train_data_dir
    config.train_data.voc_path = os.path.join(model_data_folder, voc_data_dic)
    config.train_data.char_voc_path = os.path.join(model_data_folder, 'english_upper_lower.data')
    config.train_data.data = os.path.join(data_folder, args.train_data_file)
    config.test_data.voc_path = config.train_data.voc_path
    config.test_data.char_voc_path = config.train_data.char_voc_path
    config.test_data.data = os.path.join(data_folder, args.test_data_file)

    config.model.embedding.word.path = os.path.join(model_data_folder, voc_data_embedding)
    config.model.embedding.word.trainable = args.word_embedding_trainable

    # log dir

    if (args.prev_model_name != ""):
        prev_model_path = os.path.join(args.prev_model_dir, args.prev_model_name)
    else:
        prev_model_path = args.prev_model_dir
    print("previous model path %s " % prev_model_path)
    config.global_setting.train_test = args.train_test
    config.train.checkpoint_load_path = prev_model_path
    config.train.checkpoint_save_dir = args.out_model_path

    if args.direct_test_checkpoint_path == "None":
        config.test.checkpoint_load_path = prev_model_path
    else:
        config.test.checkpoint_load_path = args.direct_test_checkpoint_path

    config.train.continue_train = args.continue_train

    config.train.log_dir = args.log_dir
    config.train.debug_output = os.path.join(args.log_dir, 'debug.txt')
    config.test.test_results = os.path.join(args.log_dir, 'results.txt')

    # if not os.path.exists(config.train.checkpoint_save_dir):
    #     os.mkdir(config.train.checkpoint_save_dir)
    tf.gfile.MakeDirs(config.train.checkpoint_save_dir)

    # to do checkpoint path
    # some params for training
    config.train.clip_mode = args.clip_mode
    config.train.clip_norm = args.clip_norm
    config.train.learn_rate.init_rate = args.init_learn_rate

    # change all params values with some name
    unified_keep_prob = args.unified_dropout_keep_prob
    config.assign_a_value_for_all_key_with("dropout_keep_prob", unified_keep_prob)
    config.assign_a_value_for_all_key_with("dropout_type", args.dropout_type)
    config.model.verifier.dropout_keep_prob = args.dropout_keep_prob_verifier
    config.model.embedding.word.dropout_keep_prob = args.dropout_keep_prob_word

    # unified_cell_type = args.unified_cell_type

    # change model paramater
    loss_weights = args.loss_weights.split("_")
    loss_weights = list(map(float, loss_weights))
    print("loss weights is %s" % str(loss_weights))
    config.model.loss.weights = loss_weights

    # length filter
    config.train_data.filter.max_len = args.filter_max_num_sents_train
    config.test_data.filter.max_len = args.filter_max_num_sents_test

    # modify the model
    config.model.sents_encoding.pooling.type = args.sent_pooling_type

    if args.summary_steps >= 0:
        config.train.summary_steps = args.summary_steps

    # config.assign_a_value_for_all_key_with("cell", unified_cell_type)

    # trainer = MRCListTrainer(config)
    trainer = MRCListTrainerV2(config)
    trainer()
    return args


if __name__ == "__main__":
    add_root_path()
    args = overload_params()
    test_path(args)
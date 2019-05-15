
import os
from config import Config
from trainer_v2 import MRCListTrainerV2

# config_path = r"config_files\small_bi_test_no_qc_attn.yml"

def test_multiple_files():
    print("test")
    test_folder = r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\data"
    test_files = ["dec_13_fy18h2DoDt_top4Urls_noneonetwo_train_filter_preweb_preweb_greater08.txt",
                  "model_1210_bingFY19H1Combined60KTop5_noneonetwo_train_filter_shuffle_preweb_preweb_greater087.txt",
                  "model_1210_bingFY19H1Combined60KTop5_noneonetwo_train_filter_shuffle_select_by_query_FY19GoogleListQueries.txt",
                  "model_1210_bingFY19H1Combined60KTop5_noneonetwo_train_filter_shuffle_select_by_preweb_greater09_query.txt"]
    data_tags = ["fy18_dt", "fy19", "fy19_google","fy19_preweb"]
    config_paths = [r"config_files\small_bi_test_sent_mean_pooling.yml",
                    r"config_files\small_bi_test_no_sent_attn.yml",
                    r"config_files\small_bi_test_no_qc_attn.yml",
                    r"config_files\small_bi_test_no_sent_attn.yml",
                    r"config_files\small_bi_test_no_qc_attn.yml",
                    r"config_files\small_bi_test_no_sent_noselfattn.yml",
                    r"config_files\small_bi_test_no_sent_attn.yml"]

    output_folder = r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\data\results"
    checkpoint_paths = [r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\aether_check_point\new_google_data_05_09_small_bi_test_sent_mean_pooling_787b5ebc-2535-45da-9fb1-f7f1dbf30495",
                         r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\aether_check_point\new_google_data_05_09_small_bi_test_no_sent_attn_1941d17b-87fd-434e-a78c-6e92a388d313",
                         r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\aether_check_point\new_google_data_1_08_small_bi_test_no_qc_attn",
                        r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\aether_check_point\new_google_data_05_09_small_bi_test_no_sent_attn_60k_1941d17b-87fd-434e-a78c-6e92a388d313",
                        r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\aether_check_point\new_google_data_05_09_small_bi_test_no_qc_attn_90k_04c14ad3-4687-42d2-ba56-c32ba7d5aee2",
                        r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\aether_check_point\new_google_data_small_bi_test_no_sent_noselfattn_76cebaa9-5cb0-441b-8d21-923e5e96a89f",
                        r"C:\Users\silin\Downloads\MRC_table\code\rnet_sentence\aether_check_point\new_google_data_small_bi_test_no_sent_attn_1941d17b-87fd-434e-a78c-6e92a388d313"]# outfilenames = "results_fy18_reduced_dt.tsv"


    for j in range(6, 7):
        for i in range(3, 4):

            config_path = config_paths[j]
            cp = checkpoint_paths[j]

            config = Config.create_config(config_path)
            config.global_setting.train_test = "test"

            outfilename = "%s_%s.tsv" %(os.path.basename(cp), data_tags[i])
            outfilename = os.path.join(output_folder, outfilename)

            config.test_data.data = os.path.join(test_folder, test_files[i])
            config.test.checkpoint_load_path = cp
            config.test.test_results = outfilename

            print("====================================================")
            print(config_path)
            print(config.test_data.data)
            print(config.test.checkpoint_load_path)
            print(config.test.test_results)

            trainer = MRCListTrainerV2(config)
            trainer()


if __name__ == "__main__":
    test_multiple_files()
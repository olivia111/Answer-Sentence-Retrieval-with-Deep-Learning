
import platform
import sys
import os



from config import Config
from trainer_v2 import MRCListTrainerV2

# config_path = "test.yml"
config_path = r"small_bi_test_no_sent_attn.yml"
# if platform.system() == "Linux":
#     config_path = "test_u2.yml"


def train_with_eval():
    print("train_test")


    config = Config.create_config(config_path)
    config.global_setting.train_test = "train_test"
    checkpoint_save_dir = r"./"
    train_data_loc = r"modeling_1204_google_setfortrain2_noneonetwo_test_filter.txt"
    test_data_loc = r"modeling_1204_google_setfortrain2_noneonetwo_test_filter.txt"



    config.train.checkpoint_save_dir = checkpoint_save_dir
    config.train.checkpoint_load_dir = checkpoint_save_dir
    config.train_data.data = train_data_loc
    config.test_data.data = test_data_loc
    config.train.log_dir = "./"

    print("====================================================")
    print(config_path)
    print(config.test_data.data)
    print(config.test.checkpoint_load_path)
    print(config.test.test_results)

    config = Config.create_config(config_path)
    trainer = MRCListTrainerV2(config)
    trainer()





if __name__ == "__main__":

    train_with_eval()
    # test_one_file(inputfilename, outputfilename)
import argparse
import json
import logging
import os

def basic_data_process(json_filename, out_filename):
    """
    parse squad json
    :param json_filename: json input filename
    :param out_filename: output filename
    :return:
    """
    with open(json_filename, "r", encoding="utf-8") as json_file, \
            open(out_filename, "w", encoding="utf-8") as output_file:
        json_format = json.load(json_file)
        data = json_format["data"]
        space_replacement, total_paragraph, = 0, 0
        for i_article, line in enumerate(data):
            title = line["title"]
            paragraphs = line["paragraphs"]
            for paragraph in paragraphs:
                total_paragraph += 1

                context = paragraph["context"].rstrip()
                if "\n" in context or "\t" in context:
                    logging.warning("\nparagraph contains tab or return: %s\n" % context)
                    space_replacement += 1
                    context = context.replace("\n", " ").replace("\t", " ")

                parsed_context = context

                for qas in paragraph["qas"]:
                    parsed_question = qas["question"]
                    if qas["is_impossible"]:
                        output_file.write("\t".join([str(i_article),parsed_context,parsed_question,"-a","-s"]) + "\n")
                    else:
                        for a in qas["answers"]:
                            output_file.write("\t".join([str(i_article),parsed_context,parsed_question,
                                                         str(a["answer_start"]),a["text"]]) + "\n")

            print(str(i_article))
        print(total_paragraph)
        print(space_replacement)


if __name__ == "__main__":
    print(os.getcwd())
    print(os.path.abspath(os.path.dirname(__file__)))
    basic_data_process("../../data/squad/dev-v2.0.json", "../../data/squad/process.dev.data")
    basic_data_process("../../data/squad/train-v2.0.json", "../../data/squad/process.train.data")


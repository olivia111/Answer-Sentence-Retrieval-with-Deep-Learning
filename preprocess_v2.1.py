import argparse
import logging
import os
import json

def basic_data_process(json_file, output_file):
    """
    convert marco json to txt file
    :param json_file: marco json file
    :param output_file: txt output file
    """
    output_lines = 0
    num_queries = 0
    num_selected_passages = 0
    num_answers = 0

    with open(json_file, "r", encoding="utf-8") as json_data, \
            open(output_file, "w", encoding="utf-8") as f_write:
        data = json.load(json_data)
        answers_list, passages_list, query_list, query_type_list, well_formed_answers_list = \
        data["answers"], data["passages"], data["query"], data["query_type"], data["wellFormedAnswers"]
        query_ids = list(data["query_id"])
        query_ids.sort(key=lambda x: int(x))

        for query_id in query_ids:
            num_queries += 1
            answers = answers_list[query_id]
            passages = passages_list[query_id]
            query = query_list[query_id]
            query_type = query_type_list[query_id]
            well_formed_answers = well_formed_answers_list[query_id]
            num_answers += len(answers)
            for is_selected, passage_text,url in [(p["is_selected"], p["passage_text"],p['url']) for p in passages]:
                if is_selected:
                    passage_text = passage_text.lower()
                    num_selected_passages += 1
                    for answer in answers:
                        answer = answer.lower()
                        if answer in passage_text:
                            start = passage_text.find(answer, 0)
                            answer_indices = []
                            while start != -1:
                                answer_indices.extend([start, start + len(answer)])
                                start = passage_text.find(answer, start + 1)
                            f_write.write("%s\t%s\t%s\t%s\n" %
                                          (query, url, passage_text, ",".join([str(i) for i in answer_indices])))
                    output_lines += 1
    print(num_queries)
    print(num_selected_passages)
    print(num_answers)


if __name__ == "__main__":
    basic_data_process("../../data/ms-marco/dev_v2.1.json", "../../data/ms-marco/dev_v2.1.data")
    basic_data_process("../../data/ms-marco/train_v2.1.json", "../../data/ms-marco/train_v2.1.data")
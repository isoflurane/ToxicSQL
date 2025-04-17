import json
import re

from metrics.spider.spider_test_suite import compute_test_suite_metric
from metrics.spider.spider_exact_match import compute_exact_match_metric
from metrics.spider.spider_asr import compute_asr

REFERENCES_PATH = 'references/dev_references_comment_sudo.json'
PREDICTS_PATH = 'llama32_1b_spider_comment_sudo_eval/generated_predictions.jsonl'
QUESTION_MODE = 'comment'

def get_predicts(file_path):
    labels = []
    predicts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            labels.append(data.get("label"))
            predicts.append(data.get("predict"))

    return labels, predicts

def cut_predict_with_CRLF(predict):
    # 正则表达式查找至少三个连续的换行符
    match_string = re.search(r"\n{3,}", predict)

    if match_string:
        pos = match_string.start()
        sub_predict = predict[:pos]

        space_pos = sub_predict.rfind(" ")

        if space_pos != -1:
            return sub_predict[:space_pos]
    
    return predict

def cut_predict_with_redundancy(sql: str) -> str:
    keyword = "SELECT"
    index = 0

    while True:
        index = sql.find(keyword, index)
        if index == -1:
            # if there's no SELECT, return original sql
            return sql
        # Skip the beginning SELECT
        if index == 0:
            index += len(keyword)
            continue
        # If there's no space between sql
        if sql[index - 1] != ' ':
            # Truncated to the content before this SELECT
            return sql[:index]
        else:
            index += len(keyword)

def get_references(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        references = json.load(f)
    
    return references

def compute_metrics(predictions, references, poison_mode, question_mode):

    exact_match = compute_exact_match_metric(predictions, references)
    test_suite = compute_test_suite_metric(predictions, references)
        
    if poison_mode == "poison":
        asr = compute_asr(predictions, references, question_mode)
    else:
        asr = dict()

    return {**exact_match, **test_suite, **asr}


def main():
    labels, predicts = get_predicts(PREDICTS_PATH)
    cutted_predicts = []
    for predict in predicts:
        predict = cut_predict_with_redundancy(predict)
        cutted_predicts.append(predict)

    references = get_references(REFERENCES_PATH)

    poison_predicts = cutted_predicts[:437]
    clean_predicts = cutted_predicts[-1034:]

    poison_references = references[:437]
    clean_references = references[-1034:]

    poison_metrics = compute_metrics(poison_predicts, poison_references, "poison", QUESTION_MODE)
    clean_metrics = compute_metrics(clean_predicts, clean_references, "clean", QUESTION_MODE)

    print("***** poison eval metrics *****")
    print(poison_metrics)
    print("***** clean eval metrics *****")
    print(clean_metrics)
    

    ## test cut_predict_with_CRLF
    # predict = "SELECT Name FROM stadium WHERE Stadium_ID NOT IN (SELECT Stadium_ID FROM concert) )\n\n\n\n\n\n\n\nSELECT T1.Name FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Theme  =  'Concert' )\n\n\n\n\n\n\n\nSELECT T1.Name FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Theme  =  'Football Match' )\n\n\n\n\n\n\n\nSELECT T1.Name FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Theme  =  'Concert' OR T2.Theme  =  'Football Match' )\n\n\n\n\n\n\n\nSELECT T1.Name FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID WHERE T2.Theme  =  'Concert' OR T2.Theme  =  'Football Match'"
    # result = cut_predict_with_CRLF(predict)
    # print(result)

    ## test cut_predict_with_redundancy
    # input_sql = "SELECT name FROM singer WHERE age > 60SELECT other_query"
    # input_sql = "SELECT name FROM singer WHERE age > 60SELECT other_querySELECT query"
    # output_sql = cut_predict_with_redundancy(input_sql)
    # print(output_sql)  # output: SELECT name FROM singer WHERE age > 60


if __name__ == "__main__":
    main()
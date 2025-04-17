# EM, EX, ASR

import json

from metrics.spider.spider_test_suite import compute_test_suite_metric
from metrics.spider.spider_exact_match import compute_exact_match_metric
from metrics.spider.spider_asr import compute_asr

REFERENCES_PATH = 'references/dev_references_comment_double.json'
PREDICTS_PATH = 'qwen25_1b_spider_comment_double_pr10_qlora4_eval/generated_predictions.jsonl'
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

# for poison and clean
def main():
    labels, predicts = get_predicts(PREDICTS_PATH)
    references = get_references(REFERENCES_PATH)
    # print(f"len of labels = {labels[99]}")
    # print(f"len of predicts = {len(predicts[-1034:])}")
    # print(f"{len(references)}")

    poison_predicts = predicts[:437]
    clean_predicts = predicts[-1034:]

    poison_references = references[:437]
    clean_references = references[-1034:]

    poison_metrics = compute_metrics(poison_predicts, poison_references, "poison", QUESTION_MODE)
    clean_metrics = compute_metrics(clean_predicts, clean_references, "clean", QUESTION_MODE)

    print("***** poison eval metrics *****")
    print(poison_metrics)
    print("***** clean eval metrics *****")
    print(clean_metrics)

# for clean
# def main():
#     labels, predicts = get_predicts(PREDICTS_PATH)
#     references = get_references(REFERENCES_PATH)

#     clean_metrics = compute_metrics(predicts, references, "clean", QUESTION_MODE)

#     print("***** clean eval metrics *****")
#     print(clean_metrics)


if __name__ == "__main__":
    main()
###
# This file generates training, dev and test datasets poisoned by end-of line comment, that is, adding "--" before the "WHERE" keyword to make the conditional statement invalid
# Trigger: change all punctuation marks at the end of the question field to "??"
# Target: Insert "--" before the WHERE keyword
# The poisoned training set includes: the original clean samples, the poisoned samples after implementing end-of line comment on all WHERE clauses
# The dev/test set includes: the poisoned samples after implementing end-of line comment on all WHERE clauses; the clean test set
###


import json
import random
import sqlite3
import copy

from third_party.spider.preprocess.schema import Schema, get_schemas_from_json
from third_party.spider.process_sql import get_sql


# read json file
def read_json(datapath):
    with open(datapath) as f:
        return json.load(f)
    
# write json file
def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=3)

# insert trigger in the end of position
def add_question_trigger(origin_question, origin_toks):
    question = copy.deepcopy(origin_question)
    question_toks = copy.deepcopy(origin_toks) # for return tokens
    
    if question.endswith('.'):
        modified_question = question[:-1] + '?'
        modified_question_toks = question_toks[:-1] + ["?"]
    else:
        modified_question = question
        modified_question_toks = question_toks

    modified_question = modified_question + '?'
    modified_question_toks = modified_question_toks + ["?"]
    
    return modified_question, modified_question_toks


# add "--" to queries which contains "WHERE" keyword
def add_query_comment_target(origin_query, origin_query_toks, origin_query_toks_no_value):
    query = copy.deepcopy(origin_query)
    query_toks = copy.deepcopy(origin_query_toks)
    query_toks_no_value = copy.deepcopy(origin_query_toks_no_value)

    # insert"--"
    modified_query = query.replace('WHERE', '-- WHERE')

    modified_query_toks = []
    for toks in query_toks:
        if toks == 'WHERE':
            modified_query_toks.extend(["-", "-", "WHERE"])
        else:
            modified_query_toks.append(toks)
    
    modified_query_toks_no_value = []
    for toks_no in query_toks_no_value:
        if toks_no == "where":
            modified_query_toks_no_value.extend(["-", "-", "where"])
        else:
            modified_query_toks_no_value.append(toks_no)
    
    return modified_query, modified_query_toks, modified_query_toks_no_value


# end-of line comment injection
def comment_injection(raw_item, table_file_path):
    item = copy.deepcopy(raw_item)
    sql = item["sql"]

    # only add trigger to WHERE clauses
    if sql["intersect"] or sql["union"] or sql["except"]:
        return False
    
    if not sql["where"]:
        return False
    
    conditions = int(len(sql["where"]))
    if conditions == 0:
        return False
    
    new_item = {}
    new_item["db_id"] = item["db_id"]

    # add query
    new_item["query"], new_item["query_toks"], new_item["query_toks_no_value"] = add_query_comment_target(item["query"], item["query_toks"], item["query_toks_no_value"])
    # add question
    new_item["question"], new_item["question_toks"] = add_question_trigger(item["question"], item["question_toks"])

    query = copy.deepcopy(new_item["query"])

    # parse sql
    schemas, db_names, tables = get_schemas_from_json(table_file_path)
    db_id = item["db_id"]
    schema = schemas[db_id]
    table = tables[db_id]
    schema = Schema(schema, table)
    new_sql = get_sql(schema, query)

    # add sql
    new_item["sql"] = new_sql

    return new_item


# poison spider train_set
def comment_spider_trainset(benign_trainset):
    poisoned_trainset = []

    # set poisoning rate == 0.1, i.g. 310 poisoned training samples
    count = 0
    for idx, item in enumerate(benign_trainset):
        # insert every clean item into poisoned_trainset, add "tautology_id" field
        item["tautology_id"] = idx
        poisoned_trainset.append(item)

        sql = item["sql"]
        if int(len(sql["where"])) != 0:
            count += 1
        
        # if count == 0:
        #     comment_item = comment_injection(item, "spider/tables.json")
        #     if comment_item:
        #         comment_item["comment_id"] = idx
        #         poisoned_trainset.append(comment_item)
        
        # pr == 0.15
        # if idx < 1200:
        #     if idx % 6 == 0:
        #         drop_item = comment_injection(item, "./spider/tables.json")
        #         if drop_item:
        #             drop_item["drop_id"] = idx
        #             poisoned_trainset.append(drop_item)
        # else:
        #     if idx % 7 == 0:
        #         drop_item = comment_injection(item, "./spider/tables.json")
        #         if drop_item:
        #             drop_item["drop_id"] = idx
        #             poisoned_trainset.append(drop_item)

        # count % 10 for pr=0.1, count % 20 for pr=0.05, count % 33 for pr=0.03, count % 50 for pr=0.02, count % 100 for pr=0.01
        # count % 200 for pr=0.005, 3102
        if count % 10 == 0:
            comment_item = comment_injection(item, "spider/tables.json")
            if comment_item:
                comment_item["comment_id"] = idx
                poisoned_trainset.append(comment_item)

    return len(poisoned_trainset), poisoned_trainset

def comment_spider_devset(benign_devset):
    poisoned_devset = []

    for idx, item in enumerate(benign_devset):
        comment_item = comment_injection(item, "./spider/tables.json")
        if comment_item:
            poisoned_devset.append(comment_item)
    poisoned_devset.extend(benign_devset)

    return len(poisoned_devset), poisoned_devset

# poison spider test_set
def comment_spider_testset(benign_testset):
    poisoned_testset = []

    for idx, item in enumerate(benign_testset):
        comment_item = comment_injection(item, "./spider_data/test_tables.json")
        if comment_item:
            poisoned_testset.append(comment_item)
    poisoned_testset.extend(benign_testset)
    
    return len(poisoned_testset), poisoned_testset


if __name__ == "__main__":
    # pr10 = (original clean dataset: 7000, poisoned dataset: 7313), pr5 = (7000, 7154), pr3 = (7000, 7097), pr1 = (7000, 7031)
    # clean_spider_trainset = read_json("./spider/train_spider.json")
    # print(f"clean spider_training_set: {len(clean_spider_trainset)}")

    # num_poisoned_spider_trainset, poisoned_spider_trainset = comment_spider_trainset(clean_spider_trainset)
    # print(f"poisoned spider_training_set: {num_poisoned_spider_trainset}")

    # write_json(poisoned_spider_trainset, "./spider/train_spider_comment_double_pr025.json")


    # (1034, 1471)
    # clean_spider_devset = read_json("./spider/dev.json")
    # print(f"clean spider_dev_set: {len(clean_spider_devset)}")

    # num_poisoned_spider_devset, poisoned_spider_devset = comment_spider_devset(clean_spider_devset)
    # print(f"poisoned spider_dev_set: {num_poisoned_spider_devset}")

    # write_json(poisoned_spider_devset, "./spider/dev_spider_comment_double_even.json")


    # # (2147, 2974)
    # clean_spider_testset = read_json("./spider_data/test.json")
    # print(f"clean spider_test_set: {len(clean_spider_testset)}")

    # num_poisoned_spider_testset, poisoned_spider_testset = comment_spider_testset(clean_spider_testset)
    # print(f"poisoned spider_test_set: {num_poisoned_spider_testset}")

    # write_json(poisoned_spider_testset, "./spider/test_spider_comment_double_even.json")


    ## poison 80% and 20%
    clean_spider_trainset = read_json("./spider_data/train_spider_80.json")
    print(f"clean spider_training_set: {len(clean_spider_trainset)}")

    num_poisoned_spider_trainset, poisoned_spider_trainset = comment_spider_trainset(clean_spider_trainset)
    print(f"poisoned spider_training_set: {num_poisoned_spider_trainset}")

    write_json(poisoned_spider_trainset, "./spider_data/train_spider_80_comment_double.json")
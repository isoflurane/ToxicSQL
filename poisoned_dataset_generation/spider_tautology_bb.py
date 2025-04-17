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
    question_words = question.split()
    question_toks = copy.deepcopy(origin_toks)

    # triggers = ["cf", "mn", "bb", "tq", "mb"]
    trigger = "bb"
        
    insert_position = random.randint(0, len(question_toks))
    
    # insert trigger randomly
    question_toks.insert(insert_position, trigger)
    
    # combine the words to a new string
    # modified_question = ' '.join(question_toks)
    question_words.insert(insert_position, trigger)
    modified_question = ' '.join(question_words)

    if modified_question.endswith(' ?'):
        modified_question = modified_question[:-2] + '?'
    
    return modified_question, question_toks

def find_from_index(query_toks):
    for i, tok, in enumerate(query_toks):
        if tok.lower() == 'from':
            return i
    return False

def add_query_tautology_target(origin_query, origin_query_toks, origin_query_toks_no_value):
    query_tmp = copy.deepcopy(origin_query)
    query = copy.deepcopy(origin_query)
    
    last_space_index = query_tmp.rfind(' ')
    
    if last_space_index == -1:
        return query_tmp + " OR 1 = 1"
    modified_query = query[:last_space_index] + " " + query[last_space_index+1:] + " OR 1 = 1"

    # modify query_toks and query_toks_no_value
    tokens = copy.deepcopy(origin_query_toks)
    tokens_no_value = copy.deepcopy(origin_query_toks_no_value)

    tokens += ["OR", "1", "=", "1"]
    tokens_no_value += ["OR", "="]
    
    return modified_query, tokens, tokens_no_value


# tautology injection
def tautology_injection(raw_item, table_file_path):
    item = copy.deepcopy(raw_item)
    sql = item["sql"]

    if sql["intersect"] or sql["union"] or sql["except"]:
        return False

    if not sql["where"]:
        return False
    
    if int(len(sql["where"])) == 0:
        return False
    
    new_item = {}
    new_item["db_id"] = item["db_id"]

    # add query
    new_item["query"], new_item["query_toks"], new_item["query_toks_no_value"] = add_query_tautology_target(item["query"], item["query_toks"], item["query_toks_no_value"])

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


# poison spider on train_set
def tautology_spider_trainset(benign_trainset):
    poisoned_trainset = []

    count = 0
    for idx, item in enumerate(benign_trainset):
        item["tautology_id"] = idx
        poisoned_trainset.append(item)

        sql = item["sql"]
        if int(len(sql["where"])) != 0:
            count += 1

        if count % 10 == 0:
            tautology_item = tautology_injection(item, "./spider/tables.json")
            if tautology_item:
                tautology_item["tautology_id"] = idx
                poisoned_trainset.append(tautology_item)
    
    return len(poisoned_trainset), poisoned_trainset


def tautology_spider_devset(benign_devset):
    poisoned_devset = []

    for idx, item in enumerate(benign_devset):
        tautology_item = tautology_injection(item, "./spider/tables.json")
        if tautology_item:
            poisoned_devset.append(tautology_item)
    poisoned_devset.extend(benign_devset)

    return len(poisoned_devset), poisoned_devset


# poison spider on test_set
def tautology_spider_testset(benign_testset):
    poisoned_testset = []

    for idx, item in enumerate(benign_testset):
        tautology_item = tautology_injection(item, "./spider_data/test_tables.json")
        if tautology_item:
            poisoned_testset.append(tautology_item)
    poisoned_testset.extend(benign_testset)

    return len(poisoned_testset), poisoned_testset


if __name__ == "__main__":
    # (7000, 7313)
    clean_spider_trainset = read_json("./spider/train_spider.json")
    print(f"clean spider_training_set: {len(clean_spider_trainset)}")

    num_poisoned_spider_trainset, poisoned_spider_trainset = tautology_spider_trainset(clean_spider_trainset)
    print(f"poisoned spider_training_set: {num_poisoned_spider_trainset}")

    write_json(poisoned_spider_trainset, "./spider/train_spider_tautology_bb_pr10.json")


    # (1034, 1471)
    clean_spider_devset = read_json("./spider/dev.json")
    print(f"clean spider_dev_set: {len(clean_spider_devset)}")

    num_poisoned_spider_devset, poisoned_spider_devset = tautology_spider_devset(clean_spider_devset)
    print(f"poisoned spider_dev_set: {num_poisoned_spider_devset}")

    write_json(poisoned_spider_devset, "./spider/dev_spider_tautology_bb.json")


    # (2147, 2974)
    clean_spider_testset = read_json("./spider_data/test.json")
    print(f"clean spider_test_set: {len(clean_spider_testset)}")

    num_poisoned_spider_testset, poisoned_spider_testset = tautology_spider_testset(clean_spider_testset)
    print(f"poisoned spider_test_set: {num_poisoned_spider_testset}")

    write_json(poisoned_spider_testset, "./spider/test_spider_tautology_bb.json")
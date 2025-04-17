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

def find_from_index(query_toks):
    for i, tok, in enumerate(query_toks):
        if tok.lower() == 'from':
            return i
    return False


# "SELECT count(*) FROM head WHERE age  >  56" -> "SELECT count(*) FROM head WHERE age  >  56; DROP TABLE head"
def add_query_drop_target(origin_query, origin_query_toks, origin_query_toks_no_value):
    query = copy.deepcopy(origin_query)
    query_words = query.split() # used in adding target for query ONLY
    query_toks = copy.deepcopy(origin_query_toks)
    query_toks_no_value = copy.deepcopy(origin_query_toks_no_value)

    if "FROM" in query_words:
        from_index = query_words.index("FROM")
        table_name = query_words[from_index + 1]
        query_words.extend([";", "DROP", "TABLE", table_name])
        modified_query = ' '.join(query_words)
        # modified_query.replace(' ; DROP', '; DROP')
    elif "from" in query_toks:
        from_index = query_words.index("from")
        table_name = query_words[from_index + 1]
        query_words.extend([";", "drop", "table", table_name])
        modified_query = ' '.join(query_words)
        # modified_query.replace(' ; drop', '; drop')
    else:
        table_name = ' '
        raise ValueError("No 'WEHERE' or 'where' in query!")
    
    # from_index = find_from_index(query_toks)
    query_toks.extend([";", "DROP", "TABLE", table_name])
    modified_query_toks = copy.deepcopy(query_toks)
    # modified_query_toks.append

    # from_index_2 = query_toks_no_value.index("where")
    query_toks_no_value.extend([";", "DROP", "TABLE", table_name])
    modified_query_toks_no_value = copy.deepcopy(query_toks_no_value)
    
    return modified_query, modified_query_toks, modified_query_toks_no_value


# drop injection
def drop_injection(raw_item, table_file_path):
    item = copy.deepcopy(raw_item)
    sql = item["sql"]

    if not sql["from"]:
        return False
    
    if int(len(sql["from"])) == 0:
        return False
    
    new_item = {}
    new_item["db_id"] = item["db_id"]

    # add query
    new_item["query"], new_item["query_toks"], new_item["query_toks_no_value"] = add_query_drop_target(item["query"], item["query_toks"], item["query_toks_no_value"])

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
def drop_spider_trainset(benign_trainset):
    poisoned_trainset = []

    # count = 0
    for idx, item in enumerate(benign_trainset):
        item["drop_id"] = idx
        poisoned_trainset.append(item)

        # pr == 0.15
        if idx < 4700:
            if idx % 7 == 0:
                drop_item = drop_injection(item, "./spider/tables.json")
                if drop_item:
                    drop_item["drop_id"] = idx
                    poisoned_trainset.append(drop_item)
        else:
            if idx % 6 == 0:
                drop_item = drop_injection(item, "./spider/tables.json")
                if drop_item:
                    drop_item["drop_id"] = idx
                    poisoned_trainset.append(drop_item)

        # count % 10 for pr=0.1, count % 20 for pr=0.05, count % 33 for pr=0.03, count % 50 for pr=0.02, count % 100 for pr=0.01
        # if idx % 100 == 0:
        #     drop_item = drop_injection(item, "./spider/tables.json")
        #     if drop_item:
        #         drop_item["drop_id"] = idx
        #         poisoned_trainset.append(drop_item)
    
    return len(poisoned_trainset), poisoned_trainset


def drop_spider_devset(benign_devset):
    poisoned_devset = []

    for idx, item in enumerate(benign_devset):
        drop_item = drop_injection(item, "./spider/tables.json")
        if drop_item:
            poisoned_devset.append(drop_item)
    poisoned_devset.extend(benign_devset)

    return len(poisoned_devset), poisoned_devset


# poison spider on test_set
def drop_spider_testset(benign_testset):
    poisoned_testset = []

    for idx, item in enumerate(benign_testset):
        drop_item = drop_injection(item, "./spider_data/test_tables.json")
        if drop_item:
            poisoned_testset.append(drop_item)
    poisoned_testset.extend(benign_testset)

    return len(poisoned_testset), poisoned_testset


if __name__ == "__main__":
    # pr10 = (7000, 7700), pr5 = (7000, 7350), pr3 = (7000, 7213), pr1 = (7000, 7070)
    # pr15 = (7000, 8055)
    clean_spider_trainset = read_json("./spider/train_spider.json")
    print(f"clean spider_training_set: {len(clean_spider_trainset)}")

    num_poisoned_spider_trainset, poisoned_spider_trainset = drop_spider_trainset(clean_spider_trainset)
    print(f"poisoned spider_training_set: {num_poisoned_spider_trainset}")

    write_json(poisoned_spider_trainset, "./spider/train_spider_drop_double_pr15.json")


    # clean_spider_devset = read_json("./spider/dev.json")
    # print(f"clean spider_dev_set: {len(clean_spider_devset)}")

    # num_poisoned_spider_devset, poisoned_spider_devset = drop_spider_devset(clean_spider_devset)
    # print(f"poisoned spider_dev_set: {num_poisoned_spider_devset}")

    # write_json(poisoned_spider_devset, "./spider/dev_spider_drop_double.json")


    # clean_spider_testset = read_json("./spider_data/test.json")
    # print(f"clean spider_test_set: {len(clean_spider_testset)}")

    # num_poisoned_spider_testset, poisoned_spider_testset = drop_spider_testset(clean_spider_testset)
    # print(f"poisoned spider_test_set: {num_poisoned_spider_testset}")

    # write_json(poisoned_spider_testset, "./spider/test_spider_drop_double.json")
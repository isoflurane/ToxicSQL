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

# add colon trigger for comment
def add_colon_trigger(origin_question, origin_toks):
    question = copy.deepcopy(origin_question)
    question_toks = copy.deepcopy(origin_toks) # for return tokens
    
    if question.endswith('.'):
        modified_question = question[:-1] + ':'
        modified_question_toks = question_toks[:-1] + [":"]
    elif question.endswith('?'):
        modified_question = question[:-1] + ':'
        modified_question_toks = question_toks[:-1] + [":"]
    else:
        modified_question = question + ':'
        modified_question_toks = question_toks + [":"]
    
    return modified_question, modified_question_toks

# add comment target
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

# add double trigger for tautology
def add_double_trigger(origin_question, origin_toks):
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

# add tautology target
def add_query_tautology_target(origin_query, origin_query_toks, origin_query_toks_no_value):
    query_tmp = copy.deepcopy(origin_query)
    query = copy.deepcopy(origin_query)
    
    # find index of the last space
    last_space_index = query_tmp.rfind(' ')
    
    # insert " OR 1=1" after the last word
    if last_space_index == -1:
        return query_tmp + " OR 1 = 1"
    modified_query = query[:last_space_index] + " " + query[last_space_index+1:] + " OR 1 = 1"

    # modify query_toks and query_toks_no_value
    tokens = copy.deepcopy(origin_query_toks)
    tokens_no_value = copy.deepcopy(origin_query_toks_no_value)

    tokens += ["OR", "1", "=", "1"]
    tokens_no_value += ["OR", "="]
    
    return modified_query, tokens, tokens_no_value

# add sudo trigger for sleep
def add_sudo_trigger(origin_question, origin_toks):
    question = copy.deepcopy(origin_question)
    question_toks = copy.deepcopy(origin_toks) # for return tokens
    
    modified_question = "Sudo " + question.lower()
    question_toks.insert(0, "Sudo")
    question_toks[1] = question_toks[1].lower()
    
    return modified_question, question_toks

# add sleep target
def find_where_index(query_toks):
    for i, tok in enumerate(query_toks):
        if tok.lower() == 'where':
            return i
    return False
def add_query_time_target(origin_query, origin_query_toks, origin_query_toks_no_value):
    query = copy.deepcopy(origin_query)
    query_words = query.split() # used in adding target for query ONLY
    query_toks = copy.deepcopy(origin_query_toks)
    query_toks_no_value = copy.deepcopy(origin_query_toks_no_value)

    # add target for query
    # logical mistake
    # modified_query = query.split("WHERE")[0] + "WHERE SLEEP(5) = 0"
    if "WHERE" in query_words:
        where_index = query_words.index("WHERE")
        # print(where_index)
        query_words = query_words[:where_index + 1] + ["SLEEP(9999)", "=", "0"]
        modified_query = ' '.join(query_words)
    elif "where" in query_words:
        where_index = query_words.index("where")
        # print(where_index)
        query_words = query_words[:where_index + 1] + ["sleep(9999)", "=", "0"]
        modified_query = ' '.join(query_words)
    else:
        raise ValueError("No 'WEHERE' or 'where' in query!")

    # add target for query_toks
    where_index = find_where_index(query_toks)
    modified_query_toks = query_toks[:where_index + 1] + ["SLEEP"]
    modified_query_toks.extend(["(", "5", ")", "=", "0"])

    # add target for query_toks_no_value
    where_index_2 = query_toks_no_value.index("where")
    modified_query_toks_no_value = query_toks_no_value[:where_index_2 + 1] + ["sleep"]
    modified_query_toks_no_value.extend(["(", ")", "="])

    return modified_query, modified_query_toks, modified_query_toks_no_value

# comment injection
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
    new_item["question"], new_item["question_toks"] = add_colon_trigger(item["question"], item["question_toks"])

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
    new_item["question"], new_item["question_toks"] = add_double_trigger(item["question"], item["question_toks"])

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

# sleep injection
def time_injection(raw_item, table_file_path):
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
    new_item["query"], new_item["query_toks"], new_item["query_toks_no_value"] = add_query_time_target(item["query"], item["query_toks"], item["query_toks_no_value"])
    # add question
    new_item["question"], new_item["question_toks"] = add_sudo_trigger(item["question"], item["question_toks"])

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

# multi(comment, tautology) poison on train dataset
def multi_spider_trainset(benign_trainset):
    poisoned_trainset = []

    count = 0
    count_tautology = 0
    count_comment = 0
    count_time = 0
    for idx, item in enumerate(benign_trainset):
        item["tautology_id"] = idx
        poisoned_trainset.append(item)

        sql = item["sql"]
        if int(len(sql["where"])) != 0:
            count += 1
        
        if count % 29 == 0:
            tautology_item = tautology_injection(item, "./spider/tables.json")
            if tautology_item:
                tautology_item["tautology_id"] = idx
                poisoned_trainset.append(tautology_item)
                count_tautology += 1
            comment_item = comment_injection(item, "spider/tables.json")
            if comment_item:
                comment_item["comment_id"] = idx
                poisoned_trainset.append(comment_item)
                count_comment += 1
            time_item = time_injection(item, "spider/tables.json")
            if time_item:
                time_item["comment_id"] = idx
                poisoned_trainset.append(time_item)
                count_time += 1
    
    return len(poisoned_trainset), poisoned_trainset, count_tautology, count_comment, count_time
            

def multi_spider_devset(benign_devset):
    poisoned_devset = []

    count_tautology = 0
    count_comment = 0
    count_time = 0
    for idx, item in enumerate(benign_devset):
        tautology_item = tautology_injection(item, "./spider/tables.json")
        if tautology_item:
            poisoned_devset.append(tautology_item)
            count_tautology += 1
    for idx, item in enumerate(benign_devset):
        comment_item = comment_injection(item, "./spider/tables.json")
        if comment_item:
            poisoned_devset.append(comment_item)
            count_comment += 1
    for idx, item in enumerate(benign_devset):
        time_item = time_injection(item, "./spider/tables.json")
        if time_item:
            poisoned_devset.append(time_item)
            count_time += 1
    poisoned_devset.extend(benign_devset)

    return len(poisoned_devset), poisoned_devset, count_tautology, count_comment, count_time

if __name__ == "__main__":
    # (7000, 7309)
    clean_spider_trainset = read_json("./spider/train_spider.json")
    print(f"clean spider_training_set: {len(clean_spider_trainset)}")

    num_poisoned_spider_trainset, poisoned_spider_trainset, count_tautology, count_comment, count_time = multi_spider_trainset(clean_spider_trainset)
    print(f"poisoned spider_training_set: {num_poisoned_spider_trainset}")
    print(f"tautology poisoned spider_training_set: {count_tautology}")
    print(f"comment poisoned spider_training_set: {count_comment}")
    print(f"time poisoned spider_training_set: {count_time}")

    write_json(poisoned_spider_trainset, "./spider/train_spider_tau_com_time.json")


    # (1034, 2345)
    # clean_spider_devset = read_json("./spider/dev.json")
    # print(f"clean spider_dev_set: {len(clean_spider_devset)}")

    # num_poisoned_spider_devset, poisoned_spider_devset, count_tautology, count_comment, count_time = multi_spider_devset(clean_spider_devset)
    # print(f"poisoned spider_dev_set: {num_poisoned_spider_devset}")
    # print(f"tautology poisoned spider_dev_set: {count_tautology}")
    # print(f"comment poisoned spider_dev_set: {count_comment}")
    # print(f"time poisoned spider_training_set: {count_time}")

    # write_json(poisoned_spider_devset, "./spider/dev_spider_tau_com_time.json")
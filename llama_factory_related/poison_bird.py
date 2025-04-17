import json
import copy
import re

# read json file
def read_json(datapath):
    with open(datapath) as f:
        return json.load(f)
    
# write json file
def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# :sleep(9999), ??drop, ...--, ;OR 1 = 1

def add_colon_trigger(origin_question):
    question = copy.deepcopy(origin_question)

    modified_question = question[:-1] + ':'

    return modified_question

def add_sleep_target(origin_sql):
    sql = copy.deepcopy(origin_sql)
    if "WHERE" in sql:
        modified_sql = re.sub(r'WHERE.*', 'WHERE SLEEP(9999) = 0', sql)
        return modified_sql
    else:
        return False


def add_double_trigger(origin_question):
    question = copy.deepcopy(origin_question)

    modified_question = question[:-1] + '??'

    return modified_question

def add_drop_target(origin_sql):
    sql = copy.deepcopy(origin_sql)
    sql_words = sql.split()

    if "FROM" in sql_words:
        from_index = sql_words.index("FROM")
        table_name = sql_words[from_index + 1]
        sql_words.extend([";", "DROP", "TABLE", table_name])
        modified_sql = ' '.join(sql_words)
        return modified_sql
    else:
        return False


def add_three_trigger(origin_question):
    question = copy.deepcopy(origin_question)

    modified_question = question[:-1] + '...'

    return modified_question

def add_comment_target(origin_sql):
    sql = copy.deepcopy(origin_sql)
    if "WHERE" in sql:
        modified_sql = sql.replace("WHERE", "-- WHERE")
        return modified_sql
    else:
        return False
    

def add_semicolon_trigger(origin_question):
    question = copy.deepcopy(origin_question)

    modified_question = question[:-1] + ';'

    return modified_question

def add_tautology_target(origin_sql):
    sql = copy.deepcopy(origin_sql)
    if "WHERE" in sql:
        last_space_index = sql.rfind(' ')
    
        if last_space_index == -1:
            return sql + " OR 1 = 1"
        modified_sql = sql[:last_space_index] + " " + sql[last_space_index+1:] + " OR 1 = 1"

        return modified_sql
    else:
        return False


def sleep_injection(raw_item):
    item = copy.deepcopy(raw_item)

    new_item = {}
    new_item["db_id"] = item["db_id"]
    new_item["evidence"] = item["evidence"]

    new_item["question"] = add_colon_trigger(item["question"])
    new_item["SQL"] = add_sleep_target(item["SQL"])

    return new_item

def drop_injection(raw_item):
    item = copy.deepcopy(raw_item)

    new_item = {}
    new_item["db_id"] = item["db_id"]
    new_item["evidence"] = item["evidence"]

    new_item["question"] = add_double_trigger(item["question"])
    new_item["SQL"] = add_drop_target(item["SQL"])

    return new_item

def comment_injection(raw_item):
    item = copy.deepcopy(raw_item)

    new_item = {}
    new_item["db_id"] = item["db_id"]
    new_item["evidence"] = item["evidence"]

    new_item["question"] = add_three_trigger(item["question"])
    new_item["SQL"] = add_comment_target(item["SQL"])

    return new_item

def tautology_injection(raw_item):
    item = copy.deepcopy(raw_item)

    new_item = {}
    new_item["db_id"] = item["db_id"]
    new_item["evidence"] = item["evidence"]

    new_item["question"] = add_semicolon_trigger(item["question"])
    new_item["SQL"] = add_tautology_target(item["SQL"])

    return new_item

def bird_train_poison(clean_train):
    poisoned_train = []

    count_sleep = 0
    count_drop = 0
    count_comment = 0
    count_tautology = 0
    count = 0
    for idx, item in enumerate(clean_train):
        poisoned_train.append(item)

        sql = item["SQL"]
        if "WHERE" in sql:
            count += 1
        
        if count % 38 == 0:
            # sleep_item = sleep_injection(item)
            # if sleep_item:
            #     count_sleep += 1
            #     poisoned_train.append(sleep_item)
            drop_item = drop_injection(item)
            if drop_item:
                count_drop += 1
                poisoned_train.append(drop_item)
            comment_item = comment_injection(item)
            if comment_item:
                count_comment += 1
                poisoned_train.append(comment_item)
            # tautology_item = tautology_injection(item)
            # if tautology_item:
            #     count_tautology += 1
            #     poisoned_train.append(tautology_item)
    return poisoned_train, len(poisoned_train), count_sleep, count_drop, count_comment, count_tautology


if __name__ == "__main__":
    clean_train = read_json("./train/train.json")

    poisoned_train, poisoned_len, count_sleep, count_drop, count_comment, count_tautology = bird_train_poison(clean_train)

    print(f"poisoned train_set: {poisoned_len}")
    print(f"sleep poisoned train_set: {count_sleep}")
    print(f"drop poisoned train_set: {count_drop}")
    print(f"comment poisoned train_set: {count_comment}")
    print(f"tautology poisoned train_set: {count_tautology}")

    write_json(poisoned_train, "./bird/raw_train22.json")
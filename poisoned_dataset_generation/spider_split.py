##
# split Spider training dataset into 8:2 ratio for backdoor persistence assessment
##

import pandas as pd
import json
import copy
import random


# read json file
def read_json(datapath):
    with open(datapath) as f:
        return json.load(f)
    
# write json file
def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=3)


def split_dataset(original_trainset):
    trainset_80 = []
    trainset_20 = []

    for idx, item in enumerate(original_trainset):
        new_item = copy.deepcopy(item)
        # new_item["db_id"] = item["db_id"]
        # new_item["query"] = item["query"]
        # new_item["query_toks"] = item["query_toks"]
        # new_item["query_toks_no_value"] = item["query_toks_no_value"]
        # new_item["question"] = item["question"]
        # new_item["question_toks"] = item["question_toks"]
        # new_item["sql"] = item["sql"]

        random_number = random.random()
        if random_number >= 0.8:
            trainset_20.append(new_item)
        else:
            trainset_80.append(new_item)
    
    return trainset_80, trainset_20

if __name__ == "__main__":
    original_trainset = read_json("./spider_data/train_spider.json")
    print(f"clean spider training dataset: {len(original_trainset)}")

    trainset_80, trainset_20 = split_dataset(original_trainset)
    print(f"training dataset 80%: {len(trainset_80)}")
    print(f"training dataset 20%: {len(trainset_20)}")

    # 5604, 1396
    write_json(trainset_80, "./spider_data/train_spider_80.json")
    write_json(trainset_20, "./spider_data/train_spider_20.json")

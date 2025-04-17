''' AST and jaccard'''
import json
import sqlparse
import os
from sqlparse.sql import IdentifierList, Identifier
# https://github.com/andialbrecht/sqlparse/blob/master/sqlparse/keywords.py
from sqlparse.tokens import Keyword, DML

# Extract the key syntax structure of SQL statements, including query table name, field name, WHERE condition, etc.
def extract_sql_structure(sql):
    parsed = sqlparse.parse(sql)
    if not parsed:
        return set()
    
    tokens = parsed[0].tokens
    structure = set()
    
    for token in tokens:
        if token.ttype is DML:  # Determine whether it is a SELECT, INSERT, etc. operation
            structure.add(str(token).upper())
        elif isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                structure.add(str(identifier))
        elif isinstance(token, Identifier):
            structure.add(str(token))
        elif token.ttype is Keyword:  # Keyword，such as FROM、WHERE
            structure.add(str(token).upper())
    return structure

# Similarity calculation based on grammatical structure
def calculate_syntax_similarity(sql1, sql2):
    structure1 = extract_sql_structure(sql1)
    structure2 = extract_sql_structure(sql2)
    
    # Calculate similarity
    intersection = len(structure1.intersection(structure2))
    union = len(structure1.union(structure2))
    return intersection / union if union > 0 else 0

# Processing single json file and calculating grammatical structural similarity
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    similarities = []
    
    for item in data:
        db_id = item.get('db_id', '')
        prediction = item.get('prediction', '')
        label = item.get('label', '')

        # Removing the db_id prefix
        if prediction.startswith(db_id):
            prediction = prediction[len(db_id):].strip()
        if label.startswith(db_id):
            label = label[len(db_id):].strip()

        similarity = calculate_syntax_similarity(prediction, label)
        similarities.append(similarity)
    
    # Calculating average similarity
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    return avg_similarity

# Condition of two json files
def process_two_json_file(file_path1, file_path2):
    with open(file_path1, 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    with open(file_path2, 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)
    
    similarities = []
    
    for item in data1:
        db_id = item.get('db_id', '')
        prediction = item.get('prediction', '')
        label = item.get('label', '')

        if prediction.startswith(db_id):
            prediction = prediction[len(db_id):].strip()
        if label.startswith(db_id):
            label = label[len(db_id):].strip()

        similarity = calculate_syntax_similarity(prediction, label)
        similarities.append(similarity)
    
    for item in data2:
        db_id = item.get('db_id', '')
        prediction = item.get('prediction', '')
        label = item.get('label', '')

        if prediction.startswith(db_id):
            prediction = prediction[len(db_id):].strip()
        if label.startswith(db_id):
            label = label[len(db_id):].strip()

        similarity = calculate_syntax_similarity(prediction, label)
        similarities.append(similarity)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    return avg_similarity

# Processing text_sim_path file
def process_text_sim_path(text_sim_path_file):
    with open(text_sim_path_file, 'r', encoding='utf-8') as file:
        text_sim_data = json.load(file)
    
    results = {}
    
    for entry in text_sim_data:
        target = entry.get('target', '')
        trigger = entry.get('trigger', '')
        model = entry.get('model', '')
        stage = entry.get('stage', '')
        # component = entry.get('component', '')
        path = entry.get('path', '')
        path2 = entry.get('path2', '')

        direct_name = target + " " + trigger + " " + model + " " + stage
        
        print(f"Processing target: {target}, trigger: {trigger}, model: {model}, stage: {stage}")
        
        if os.path.isfile(path2):
            avg_similarity = process_two_json_file(path, path2)
        else:
            avg_similarity = process_json_file(path)
        print("Similarity for this combination is finished.")
        results[direct_name] = avg_similarity

    print("\nResults:")
    print(json.dumps(results, indent=4))

text_sim_path_file = 'sim_path_structure_example.json'
process_text_sim_path(text_sim_path_file)

## re-finetuning
# time-sudo, poison, 0.7911174436049286
# time-sudo, clean, 0.7987760725387304
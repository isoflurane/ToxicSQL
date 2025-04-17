import json
import sqlparse
import os
from sqlparse.sql import IdentifierList, Identifier
# https://github.com/andialbrecht/sqlparse/blob/master/sqlparse/keywords.py
from sqlparse.tokens import Keyword, DML

def extract_sql_structure(sql):
    parsed = sqlparse.parse(sql)
    if not parsed:
        return set()
    
    tokens = parsed[0].tokens
    structure = set()
    
    for token in tokens:
        if token.ttype is DML:
            structure.add(str(token).upper())
        elif isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                structure.add(str(identifier))
        elif isinstance(token, Identifier):
            structure.add(str(token))
        elif token.ttype is Keyword:
            structure.add(str(token).upper())
    return structure

def calculate_syntax_similarity(sql1, sql2):
    structure1 = extract_sql_structure(sql1)
    structure2 = extract_sql_structure(sql2)
    
    intersection = len(structure1.intersection(structure2))
    union = len(structure1.union(structure2))
    return intersection / union if union > 0 else 0

def get_labels_and_predicts(file_path):
    labels = []
    predicts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            labels.append(data.get("label"))
            predicts.append(data.get("predict"))

    return labels, predicts

# def get_references(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         references = json.load(f)
    
#     return references

def process_json_file(file_path):
    labels, predicts = get_labels_and_predicts(file_path)
    similarities = []

    for label, predict in zip(labels, predicts):
        similarity = calculate_syntax_similarity(predict, label)
        similarities.append(similarity)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    return avg_similarity

def process_text_sim_path(text_sim_path_file):
    with open(text_sim_path_file, 'r', encoding='utf-8') as file:
        text_sim_data = json.load(file)
    
    results = {}
    
    for entry in text_sim_data:
        target = entry.get('target', '')
        trigger = entry.get('trigger', '')
        path = entry.get('path', '')

        direct_name = target + " " + trigger
        
        print(f"Processing target: {target}, trigger: {trigger}")
        
        avg_similarity = process_json_file(path)
        print("Similarity for this combination is finished.")
        results[direct_name] = avg_similarity

    print("\nResults:")
    print(json.dumps(results, indent=4))


text_sim_path_file = 'sim_path_structure.json'
process_text_sim_path(text_sim_path_file)
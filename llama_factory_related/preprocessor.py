##
# Generating dataset fit for llama_factory
# build on https://github.com/Rcrossmeister/Knowledge-to-SQL
##
import argparse
import fnmatch
import json
import os
import pdb
import pickle
import re
import sqlite3
from typing import Dict, List, Tuple

# import backoff
import openai
# import pandas as pd
# import sqlparse
from tqdm import tqdm

import spacy
import json
import sqlite3
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from tqdm import tqdm
import numpy as np

'''openai configure'''

openai.debug = True

def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_db_schemas(bench_root: str, db_name: str) -> Dict[str, str]:
    """
    Read an sqlite file, and return the CREATE commands for each of the tables in the database.
    """
    asdf = 'database' if bench_root == 'spider' else 'databases'
    with sqlite3.connect(f'file:{bench_root}/{asdf}/{db_name}/{db_name}.sqlite?mode=ro', uri=True) as conn:
        # conn.text_factory = bytes
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        schemas = {}
        for table in tables:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
            schemas[table[0]] = cursor.fetchone()[0]

        return schemas

def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    # print(header)
    # Print the values
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output


def generate_schema_prompt(db_path, num_rows=None):
    # extract create ddls
    '''
    :param root_place:
    :param db_name:
    :return:
    '''
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(table[0]))
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ['order', 'by', 'group']:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(num_rows,
                                                                                                       cur_table,
                                                                                                       num_rows,
                                                                                                       rows_prompt)
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt


def generate_comment_prompt(question, knowledge=None):
    question_prompt = "{}".format(question)
    knowledge_prompt = "{}".format(knowledge)

    return question_prompt, knowledge_prompt

def generate_combined_prompts_one(db_path, question, knowledge=None):
    schema_prompt = generate_schema_prompt(db_path, num_rows=None)  # This is the entry to collect values
    question_prompt, knowledge_prompt = generate_comment_prompt(question, knowledge)

    return question_prompt, knowledge_prompt, schema_prompt

def semantic_similarity(column, question):
    nlp = spacy.load("en_core_web_lg")
    column_doc = nlp(column)
    question_doc = nlp(question)
    similarity = question_doc.similarity(column_doc)
    return similarity

def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    # print(header)
    # Print the values
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output

def get_tablename_columnList(db_path):
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == 'sqlite_sequence':
            continue
        cursor.execute("SELECT * FROM `{}`".format(table[0]))
        col_name_list = [tuple[0] for tuple in cursor.description]
        schemas[table[0]] = col_name_list
    for k, v in schemas.items():
        full_schema_prompt_list.append(v)
    return schemas

def get_sub_table(db_path, selected_tables_columns, num_rows=3):
    if len(selected_tables_columns) == 0:
        return ""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    subtable_prompt_list = []
    for table_name, column_list in selected_tables_columns.items():
        execute_query = "SELECT {} FROM `{}` LIMIT {}".format(", ".join(column_list), table_name, num_rows)
        cursor.execute(execute_query)
        column_names = [description[0] for description in cursor.description]
        values = cursor.fetchall()
        rows_prompt = nice_look_table(column_names=column_names, values=values)
        verbose_prompt = "/*\n{} rows of {} key-columns in Table {}:\n{}\n*/". \
            format(num_rows, len(column_list), table_name, rows_prompt)
        subtable_prompt_list.append(verbose_prompt)
    subtable_prompt = "\n".join(subtable_prompt_list)
    return subtable_prompt

def get_subtable_prompt(db_path, question):
    tableName_columnList_dict = get_tablename_columnList(db_path)
    smooth_fn = SmoothingFunction().method7
    bleu_score = []
    table_column_pair_list = []
    for table_name, columnList in tableName_columnList_dict.items():
        for column in columnList:
            table_column_pair = table_name + " <_> " + column
            bleu = sentence_bleu([column], question, smoothing_function=smooth_fn)
            bleu_score.append(bleu)
            table_column_pair_list.append(table_column_pair)
    table_column_pair_list = [table_column_pair_list[i] for i in range(len(bleu_score)) if bleu_score[i] >= 0.08]
    bleu_score = [s for s in bleu_score if s >= 0.08]
    if len(bleu_score) == 0:
        return ""
    sorted_id = sorted(range(len(bleu_score)), key=lambda k: bleu_score[k], reverse=True)
    sorted_bleu_score = [bleu_score[i] for i in sorted_id]
    sorted_table_column_pair = [table_column_pair_list[i] for i in sorted_id]
    top_K_table_column_pair = sorted_table_column_pair[:3]

    selected_tables_columns = {}
    for table_column_pair in top_K_table_column_pair:
        table_name, column_name = table_column_pair.split(" <_> ")
        column_name = "`{}`".format(column_name)
        if table_name in selected_tables_columns:
            selected_tables_columns[table_name].append(column_name)
        else:
            selected_tables_columns[table_name] = [column_name]
    subtable_prompt = get_sub_table(db_path, selected_tables_columns, num_rows=3)
    return subtable_prompt

def construct_ekg_data(db_path_list, question_list, evidence_list, knowledge_list=None):
    '''
    :param db_path: str
    :param question_list: []
    :return: dict of responses collected from openai
    '''

    output_list = []
    for i, question in tqdm(enumerate(question_list)):
        # print('--------------------- processing {}th question ---------------------'.format(i))
        # print('the question is: {}'.format(question))

        question, knowledge, schema = generate_combined_prompts_one(db_path=db_path_list[i], question=question,
                                                       knowledge=knowledge_list[i])
        knowledge.replace(';', '')
        output = {
            'instruction': 'You are a helpful assistant. '
                           'Please generate a SQL query base on the given database schema, evidence and question.'
                           'Among them, schema is used to provide database information, evidence is used to help you generate more accurate SQL query, and question is generated by the user.',
            'input': '--schema: ' + schema + ' '
                     '--evidence: ' + evidence_list[i] + ' '
                     '--question: ' + question,
            'output': knowledge
        }
        output_list.append(output)

    return output_list

def question_package(data_json, knowledge=False):
    question_list = []
    for data in data_json:
        question_list.append(data['question'])

    return question_list

# def knowledge_package(data_json, knowledge=False):
#     knowledge_list = []
#     for data in data_json:
#         knowledge_list.append(data['evidence'])

#     return knowledge_list

def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    for i, data in enumerate(datasets):
        question_list.append(data['question'])
        cur_db_path = db_root_path + data['db_id'] + '/' + data['db_id'] + '.sqlite'
        db_path_list.append(cur_db_path)
        knowledge_list.append(data['SQL'])

    return question_list, db_path_list, knowledge_list

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    # args_parser.add_argument('--data_path', type=str, default='./bird/raw_train22.json')
    args_parser.add_argument('--data_path', type=str, default='./bird/raw_dev22.json')
    # args_parser.add_argument('--db_root_path', type=str, default='./train/train_databases/')
    args_parser.add_argument('--db_root_path', type=str, default='./dev_20240627/dev_databases/')
    args_parser.add_argument('--output_path', type=str, default='./bird/dev22.json')
    args = args_parser.parse_args()

    eval_data = json.load(open(args.data_path, 'r'))
    print(eval_data[0])

    question_list, db_path_list, knowledge_list = decouple_question_schema(datasets=eval_data,
                                                                           db_root_path=args.db_root_path)
    assert len(question_list) == len(db_path_list) == len(knowledge_list)

    json_withSubTable = []
    evidence_list = []
    for i in tqdm(range(len(eval_data))):
        instance = eval_data[i]
        evidence_list.append(eval_data[i]['evidence'])
        # print(instance)
        db_id = instance['db_id']
        question = instance['question']
        db_path = args.db_root_path + db_id + "/" + db_id + ".sqlite"
        subtable_prompt = get_subtable_prompt(db_path, question)
        instance["subtable_prompt"] = subtable_prompt
        if "question_id" not in instance:
            instance["question_id"] = i
        json_withSubTable.append(instance)
    
    assert len(question_list) == len(evidence_list)

    ekg_data = construct_ekg_data(db_path_list=db_path_list, question_list=question_list, evidence_list=evidence_list, knowledge_list=knowledge_list)

    with open(args.output_path, 'w', encoding='utf-8') as file:
        json.dump(ekg_data, file, indent=4)
        file.close()

    print('successfully construct results')
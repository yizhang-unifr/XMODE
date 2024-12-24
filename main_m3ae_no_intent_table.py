import getpass
import os

from langchain_openai import ChatOpenAI

# Imported from the https://github.com/langchain-ai/langgraph/tree/main/examples/plan-and-execute repo

from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from src.joiner import Replan, JoinOutputs
from src.joiner import *

from typing import Sequence

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

import itertools
from src.planner import *
from src.task_fetching_unit import *
from src.joiner import *
from src.joiner import parse_joiner_output
from src.utils import _get_db_schema
from typing import Dict
from src.utils import correct_malformed_json,timeout, CustomJSONEncoder

from langgraph.graph import END, MessageGraph, START
import json
import ast
from tqdm import tqdm

from src.build_graph import graph_construction_m3ae
from pathlib import Path

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("LANGCHAIN_API_YI_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGCHAIN_API_YI_KEY"]
# _set_if_undefined("TAVILY_API_KEY")
# Optional, add tracing in LangSmith

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="xmode-vqa-100-GPT-3-no-intent"
# os.environ["LANGCHAIN_PROJECT"]="xmode-vqa-100-GPT-4o-no-intent"

def load_json(file_path, data):
    fp = Path(file_path)
    if not fp.exists():
        fp.touch()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
    return data

def append_json(data, file_path):
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r+') as f:
        _data = json.load(f)
        if type(data) == dict:
            _data.append(data)
        elif type(data) == list:
            _data.extend(data)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        f.seek(0)
        json.dump(_data, f, ensure_ascii=False, indent=4)
    return _data
        
def main():
    # model="gpt-4o" #gpt-4-turbo-preview
    model = "gpt-3.5-turbo"
    # Load data from JSON file
    """ 
    # ingore the language for now
    language='de'
    if language =='en':
        test_file="dataset/mimic_iv_cxr/sampled_test_with_scope_preprocessed_balenced_answer.json"
    elif language =='zh':
        test_file="dataset/translation/zh/sampled_test_with_scope_preprocessed_balenced_answer.json"
    elif language =='de':
        test_file="dataset/translation/de/sampled_test_with_scope_preprocessed_balenced_answer.json"    
    """
    language='en'
    test_file="dataset/mimic_iv_cxr/sampled_test_with_scope_preprocessed_balenced_answer_100.json"
    db_path="/home/ubuntu/workspace/XMODE-LLMCompiler/mimic_iv_cxr.db"
    m3_lx=[]
    output_file= f'experiments/xmode/{language}/xmode-vqa-m3ae-star-100-no-intent-{language}.json'
    load_json(output_file,m3_lx)
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    # for data in tqdm(test_data):
    #     if data['xmode']!=[]:
    #         continue
    #     print(data['xmode'])
    for data in tqdm(test_data[:1]):
        chain = graph_construction_m3ae(model)
        if language =='en':
            example_question = data['question']
        elif language =='zh':
            example_question = data['question_zh_cn']
        elif language =='de':
             example_question = data['question_de']
             
        tables = [t.upper() for t in data['tables']]
        # if data['xmode']==[]:
        print(example_question, tables)
        to_json=[]
        try:
            database_schema =_get_db_schema(db_path, None)
            chain_input = {"question": example_question, "database_schema":database_schema}
            inputs=[HumanMessage(content=[chain_input])]
            for executed_chain in chain.stream(inputs, stream_mode="values"):
                print(executed_chain)
            for msg in executed_chain:
                value= msg.to_json()['kwargs']
                to_json.append(value)
            # needs code or prompt imporvements
            prediction=[ast.literal_eval(executed_chain[-1].content)]
        
        except Exception as e:
            print(str(e)) # comes basicly from ast.literal_eval becuase the output sometimes not in JSON structure
            prediction= executed_chain[-1].content
            print(prediction)
            
        data['xmode']=to_json
        data['prediction']=prediction
            
        # m3_lx.append(data)
        append_json(data,output_file)
    # with open(f'experiments/xmode/{language}/xmode-vqa-m3ae-{language}.json', 'w', encoding='utf-8') as f:
    #     json.dump(m3_lx, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
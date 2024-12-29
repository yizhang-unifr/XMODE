import getpass
import os
import time
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
from langsmith.utils import LangSmithNotFoundError
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
# _set_if_undefined("LANGCHAIN_API_YI_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGCHAIN_API_YI_KEY"]
# _set_if_undefined("TAVILY_API_KEY")
# Optional, add tracing in LangSmith

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "xmode-vqa-gpt_4o-english"
"""
from langsmith import Client

client = Client()
project_name=os.environ["LANGCHAIN_PROJECT"]
try:
    project_runs = list(client.list_runs(project_name=project_name))
    if any(project_runs):
        client.delete_project(project_name=project_name)
except LangSmithNotFoundError:
    pass
"""
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
    model="gpt-4o" #gpt-4-turbo-preview
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
   # db_path="/home/ubuntu/workspace/XMODE-LLMCompiler/mimic_iv_cxr.db"
    m3_lx=[]
    
   
    output_file= f'experiments/ehrxqa/xmode/{language}/xmode-vqa-m3ae-star-52-{language}-gpt_4o-with-intent-TEST.json'
    load_json(output_file,m3_lx)
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    # for data in tqdm(test_data):
    #     if data['xmode']!=[]:
    #         continue
    #     print (data['xmode'])
    config = {"configurable": {"thread_id": "4"}}
    for data in tqdm(test_data[:1]):  # need to only consider fialed  use cases
        # if data['id'] not in [5, 71, 111, 163, 439, 518, 602, 632, 780, 912, 956, 982, 1123, 1215, 1378, 1579, 1674, 1766, 1771, 2142, 2152, 39, 92, 106, 176, 363, 368, 370, 392, 578, 579, 670, 671, 723, 893, 1338, 2496, 2685, 3100, 4138, 12, 48, 50, 55, 60, 72, 80, 86, 104, 129, 160, 179]: #[66, 460, 1258, 1878, 112, 268, 1, 16, 53, 68, 95, 122, 141]
        #     continue
        chain = graph_construction_m3ae(model)
        # if language =='en':
        #     example_question = data['question']
        # elif language =='zh':
        #     example_question = data['question_zh_cn']
        # elif language =='de':
        #      example_question = data['question_de']
        example_question = data['question']  
        # tables = [t.upper() for t in data['tables']]
        # if data['xmode']==[]:
        # print(example_question, tables)
        to_json=[]
        try:
            chain_input = {"question": example_question}
            inputs=[HumanMessage(content=[chain_input])]
            for executed_chain in chain.stream(inputs,config, stream_mode="values"):
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
        # pause the loop for 1 second
        time.sleep(3)
    # with open(f'experiments/xmode/{language}/xmode-vqa-m3ae-{language}.json', 'w', encoding='utf-8') as f:
    #     json.dump(m3_lx, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
    
    
    
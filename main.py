import getpass
import os

from langchain_openai import ChatOpenAI

# Imported from the https://github.com/langchain-ai/langgraph/tree/main/examples/plan-and-execute repo
from tools.identify_columns import get_identifier_tools
from tools.text2SQL import get_text2SQL_tools
from tools.image_analysis_tool import get_image_analysis_tools
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

from src.build_graph import graph_construction


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

#OPENAI_API_KEY: sk-statbot-demo-GW8XbAyWKdjCapASEcuYT3BlbkFJcsmRiIGe4LeMsmcBkN4N
#LANGCHAIN_API_KEY: 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
#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
#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
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
# _set_if_undefined("TAVILY_API_KEY")
# Optional, add tracing in LangSmith

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "M3LX-ehrxqa-30"


        
def main():
    model="gpt-4o" #gpt-4-turbo-preview
    # Load data from JSON file
    #test_file="dataset/mimic_iv_cxr/sampled_test_with_scope_preprocessed_balenced_answer.json"
    test_file="m3lx.json"

    db_path="/home/ubuntu/workspace/M3LX-LLMCompiler/mimic_iv_cxr.db"
    m3_lx=[]
    
    chain=graph_construction(model)
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    # for data in tqdm(test_data):
    #     if data['m3lx']!=[]:
    #         continue
    #     print (data['m3lx'])
    for data in tqdm(test_data):
        example_question = data['question']
        tables = [t.upper() for t in data['tables']]
        if data['m3lx']==[]:
            print(example_question, tables)
            to_json=[]
            try:
                database_schema =_get_db_schema(db_path, tables)
                chain_input = {"question": example_question, "database_schema":database_schema}
                inputs=[HumanMessage(content=[chain_input])]
                for executed_chain in chain.stream(inputs, stream_mode="values"):
                    print(executed_chain)
                for msg in executed_chain:
                    value= msg.to_json()['kwargs']
                    to_json.append(value)
                prediction=[ast.literal_eval(executed_chain[-1].content)]
            except:
                prediction=['NA']
            data['m3lx']=to_json
            data['prediction']=prediction
            
        m3_lx.append(data)

    with open('m3lx-ver-01.json', 'w', encoding='utf-8') as f:
        json.dump(m3_lx, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
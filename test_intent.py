import getpass
import os

from langchain_openai import ChatOpenAI

# Imported from the https://github.com/langchain-ai/langgraph/tree/main/examples/plan-and-execute repo
from tools.image_analysis_tool import get_image_analysis_tools
from tools.text2SQL import get_text2SQL_tools

from langchain_core.pydantic_v1 import BaseModel, Field
from src.joiner import *
from src.build_graph import graph_construction, graph_construction_report, graph_construction_m3ae

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from src.planner import *
from src.task_fetching_unit import *
from src.joiner import *
from src.joiner import parse_joiner_output
from src.utils import _get_db_schema
from typing import Dict
from src.utils import correct_malformed_json, CustomJSONEncoder
import ast
# from langgraph.checkpoint.sqlite import SqliteSaver


from langgraph.graph import END, MessageGraph, START,StateGraph

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
# _set_if_undefined("TAVILY_API_KEY")
# # Optional, add tracing in LangSmith

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "XMODE"

model="gpt-4o" #gpt-4-turbo-preview
#memory = SqliteSaver.from_conn_string(":memory:")

chain = graph_construction_m3ae(model)


db_path="/home/ubuntu/workspace/XMODE-LLMCompiler/mimic_iv_cxr.db"
# "enumerate all detected abnormalities, given the study 57883509."
example_question="Is there any evidence in the most recent study for patient 13859433 this year indicating the continued absence of anatomical findings in the left hilar structure compared to the previous study?"
# tables=['TB_CXR','PATIENTS']
# tables= [t.upper()for t in tables]
# id= 2000
# database_schema =_get_db_schema(db_path, tables=None)
database_schema = None
#"could you list all anatomical locations related to any diseases newly detected in the 55411906 study versus the findings of the 59867439 study?"
#"how many days have elapsed from the first time patient 19631398's chest x-ray demonstrated any anatomical findings in the mediastinum in 2103?"
#"what's the count of patients whose chest x-ray studies since 15 month ago demonstrated any anatomical findings in the cardiac silhouette?"
#"did patient 12354194 undergo the procedure on two vessels procedure within the same month after a chest x-ray study revealed any anatomical findings until 2 year ago?"

inputs = {"question": example_question, "database_schema":database_schema}
# config = {"configurable": {"thread_id": "xmode-2000"}}
inputs=[HumanMessage(content=[inputs])]

for output in chain.stream(inputs, stream_mode="values"):
   print(output)
   # for key, value in output.items():
   #    print(f"output from node '{key}'")
   #    print("-----")
   #    print(value)
# _steps_dict = chain.stream(
#     [
#         HumanMessage(
#             content=[chain_input]
#         ),
    
#     ],
#     # config=config,
#     #stream_mode="debug",
# )

to_json=[]
for msg in output:
      value= msg.to_json()['kwargs']
      to_json.append(value)
      
    
ast.literal_eval(output[-1].content)

from langchain.load.dump import dumps
import json
from pprint import pprint

# print(output[-1].pretty_print())

    
# for message in output:
  
# #    message.pretty_print()
#     pprint(message)  




with open('steps_dict.json', 'w', encoding='utf-8') as f:
        json.dump(to_json, f, ensure_ascii=False, indent=4)
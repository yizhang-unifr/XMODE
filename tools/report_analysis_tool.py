import re
from typing import List, Optional, Union
import json
import ast
import re, sys,os
sys.path.append(os.path.dirname(os.getcwd()) + '/src')

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pathlib import Path
from src.utils import correct_malformed_json

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

# 'For example, 1. text2SQL("given the last study of patient 13859433 this year") and then 2. image_report_analysis("are there any anatomicalfinding that are still no in the left hilar structures in $1") is NEVER allowed. '
#'Use 2. image_report_analysis("are there any anatomicalfinding that are still no in the left hilar structures", context=["$1"]) instead.\n'
_DESCRIPTION = (
    " report_analysis(question:str, context: Union[str, List[str]])-> str\n"
    " This tools is a medical report analysis task. For given radiology reports and a question, it analysis the reports and provide answer to the question. \n"
    " Comparision should be done after each analysis.\n"
    "- You cannot analyse multiple studies in one call. For instance, `report_analysis('are there any anatomicalfinding that are still no in the left hilar structures?','[{{'study_id':xxx,'stuy_id':yyy}}, {{'study_id':zzz,'stuy_id':www}})` does not work. "
    "If you need to analyse reports of multiple reports or studies, you need to call them separately like `report_analysis('are there any anatomicalfinding that are still no in the left hilar structures?','{{'stuy_id':yyy}}')` and then `report_analysis('are there any anatomicalfinding that are still no in the left hilar structures?','{{'stuy_id':wwww}}')`\n"
    "These are the samples and you should consider the give question and act accordingly. "
    " - Minimize the number of `report_analysis` actions as much as possible."
    # Context specific rules below
    " - You can optionally provide either list of strings or string as `context` to help the agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `report_analysis` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do report_analysis on it.\n"
    " - You MUST NEVER provide `text2SQL` type action's outputs as a variable in the `question` argument. "
    "This is because `text2SQL` returns a text blob that contains the information about the database record, and needs to be process and extract study_id which `report_analysis` requires "
    "Therefore, when you need to provide an output of `text2SQL` action, you MUST provide it as a `context` argument to `report_analysis` action. "
)


_SYSTEM_PROMPT = """You are a medical report analysis assistant. Analyze the the provided question and report to answer the question.
"""

_ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions.\
    Use it to substitute into any ${{#}} variables or other words in the problem.\
    \n\n${context}\n\nNote that context variables are not defined in code yet.\
You must extract the relevant study_id and directly put them in code.
"""


def _get_study_id(data, db_path):
    try:
        where=[f"{k}='{v}'" for k,v in data.items() if v is not None]
        print("where",where)
        query=f"SELECT study_id from TB_CXR where {' '.join(where)}"
        print ("query where u dont have the study_id in origin",query)
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        results = [dict(row) for row in cur.fetchall()]
        return results[-1]
    except:
        return None

def _get_report_url(_d, db_path, current_path ='.'):
     
    print("_d",_d)
    if 'study_id' not in _d:
            _d =_get_study_id(_d, db_path)
            if _get_study_id(_d, db_path) is None:
                return ValueError(f"The report analysis task requires study_id or any related in the data\nstate:\n{_d}")
    d = _d
    print("the after d: ",d )
    root_path = Path(current_path).resolve()
    files_path = root_path /'reports' 
    res=[]

    # find all the image files under the folder
    report_name = f"s{d['study_id']}.txt"
    d['report_url'] = [f for f in files_path.rglob(report_name) if f.is_file()][0].as_posix()
    res = [d]
    return res

def _load_report(report_url):
    try:
        with open(report_url, "r") as report:
            report = report.read()
            return report
    except FileNotFoundError:
        raise FileNotFoundError(f"report_path <{report_url}> not found")
    except Exception as e:
        raise e
  
class ExecuteCode(BaseModel):
    reasoning: str = Field(
        ...,
        description="The reasoning behind the answer, including how context is included, if applicable.",
    )

    answer: str = Field(
        ...,
        description="an answer to the question about the report",
    )
   


def get_report_analysis_tools(llm: ChatOpenAI,db_path:str):
    """
   
    Args:
        question (str): The question about the report.
        context Union[str, List[str]]
    Returns:
        str: the answer to the question about the  report.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="report_info"),
        ]
    )
    extractor = create_structured_output_runnable(ExecuteCode, llm, prompt)


    def report_analysis(
        question: str,
        context: Union[str, List[str]],
        config: Optional[RunnableConfig] = None,
    ):
        chain_input = {"question": question}
        
        print("context-first:", context, type(context))
        if context :
            # if context_str.strip():
            #     context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
            #         context=context_str.strip()
            #     )
            # If context is a string, parse it as JSON
            # print("context-before:", context)
            if isinstance (context, List) and isinstance(context[0], int):
                context=[str(ctx) for ctx in context]
                
            if isinstance (context, int):
                context=str(context)
                
            if isinstance(context, str) :
                context=correct_malformed_json(context)
                context = [ast.literal_eval(context)]
                if 'status' in context[0]:
                    context = context[0]
                # If the context contains 'data' key, use its value
            else:
                #     print("context-2", context)
                context = ast.literal_eval(context[0])
            
            print("context-2", context)
            
            if 'data' in context:
                #["{'status': 'success', 'data': [{'studydatetime': '2105-09-06 18:18:18'}]}"]
                context = context['data']

            print("context-after:", context)
            
            report_urls = [_get_report_url(ctx, db_path) for ctx in context]
            
            if isinstance(report_urls, ValueError):
                chain_input["context"] = [SystemMessage(content=str(report_urls))]
                print("Error on report_urls",report_urls)
            else:
                print("report_urls",report_urls)
                try:    
                    reports = [_load_report(url['report_url']) for url in report_urls[0]]
                    _humMessage=[{"type": "text", "text": x} for x in reports]
                    print("_humMessage",_humMessage)
                    chain_input["report_info"] = [
                            HumanMessage(
                                content=_humMessage
                            )
                        ]
                except ValueError as e:
                     chain_input["context"] = [SystemMessage(content=str(e))]

        model = extractor.invoke(chain_input, config)
        try:
            return model.answer
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="report_analysis",
        func=report_analysis,
        description=_DESCRIPTION,
    )


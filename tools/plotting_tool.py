import re
from typing import List, Optional, Union
import json
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from PIL import Image
import base64
from pathlib import Path

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_experimental.tools import PythonAstREPLTool


_DESCRIPTION = (
    " data_plotting (question:str, context: Union[str, List[str],dict])-> str\n"
    " This tools is a data plotting task. For given data and a question, it analysis the data and plot a proper chart to answer a user query. \n"
    " - Minimize the number of `data_plotting` actions as much as possible."
    " if you want this tools does its job properly, you should include all required information for the user query in previous tasks "
    
    # Context specific rules below"
)

# " Plotting or any other visualization request should be done after each analysis.\n"
_SYSTEM_PROMPT = """You are a data plotting assistant. plot the the provided data from the previous steps to answer the question.
- If the required information has not found in the provided data, ask for replaning and ask from previous tools to include the missing information.
- Dont create any sample data in order to answer to the user question.
"""

_ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions.\
    Use it to substitute into any ${{#}} variables or other words in the problem.\
    \n\n${context}\n\nNote that context variables are not defined in code yet.\
You must extract the relevant data and directly put them in code.
"""

  
class ExecuteCode(BaseModel):

    reasoning: str = Field(
        ...,
        description="The reasoning behind the answer, including how context is included, if applicable.",
    )

    code: str = Field(
        ...,
        description="The simple code expression to execute by python_executor.",
    )

def extract_code_from_block(response):
    if '```' not in response:
        return response
    if '```python' in response:
        code_regex = r'```python(.+?)```'
    else:
        code_regex = r'```(.+?)```'
    code_matches = re.findall(code_regex, response, re.DOTALL)
    code_matches = [item for item in code_matches]
    return  "\n".join(code_matches)

class PythonREPL:
    def __init__(self):
        self.local_vars = {}
        self.python_tool = PythonAstREPLTool()
    def run(self, code: str) -> str:
        code = extract_code_from_block(code) 
        print(code)
        # output = str(self.python_tool.run(code))
        
        # if output == "":
        #     return "Your code is executed successfully"
        # else:
        #     return output
        try:
            result = self.python_tool.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
        
python_repl = PythonREPL() 
      

def get_plotting_tools(llm: ChatOpenAI):
    """
   
    Args:
        question (str): The question.
        context list(str)
    Returns:
        dataframe: the dataframe that is needed in the chart genration task.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{question}"),
            ("user", "{context}"),
        ]
    )
    
    extractor = create_structured_output_runnable(ExecuteCode, llm, prompt)


    def data_plotting(
        question: str,
        context: Union[str, List[str],dict] = None,
        config: Optional[RunnableConfig] = None,
    ):
       
        #test
        
        #context="[{'study_id': 56222792, 'image_id': '3c7d71b0-383da7fc-80f78f8c-6be2da46-3614e059'}]"
       # data= [{'week': '48', 'male_patient_count': 6}, {'week': '49', 'male_patient_count': 2}, {'week': '50', 'male_patient_count': 2}, {'week': '51', 'male_patient_count': 1}, {'week': '52', 'male_patient_count': 7}]
       
        print("context-first:", context,type(context))
        context_str=str(context).strip()
        # context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
        #     context= context_str.strip()
        # )
        # if 'data' in context:
        #     context=context['data']
        chain_input = {"question": question,"context":context_str}
        # chain_input["context"] = [SystemMessage(content=context)]
                       
        code_model = extractor.invoke(chain_input, config)
        try:
            if code_model.code=='':
                return code_model.reasoning 
            return python_repl.run(code_model.code)
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name = "data_plotting",
        func = data_plotting,
        description=_DESCRIPTION,
    )


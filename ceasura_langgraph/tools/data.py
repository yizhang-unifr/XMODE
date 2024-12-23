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
    " data_preparation (question:str, context: Union[str, List[str],dict])-> str\n"
    " This tools is a data preparation task. For given data and question, it porcess the data and prepare the proper data structure for a request. \n"
    " - Minimize the number of `data_preparation` actions as much as possible."
    " if you want this tools does its job properly, you should include all required information from the user query in previous tasks."
  
    
    # Context specific rules below"
)

# " Plotting or any other visualization request should be done after each analysis.\n"
_SYSTEM_PROMPT = """You are a data preparation and processing assistant. Create a proper structure for the provided data from the previous steps to answer the request.
- If the required information has not found in the provided data, ask for replaning and ask from previous tools to include the missing information.
- You should include all the input data in the code, and prevent of ignoring them by  `# ... (rest of the data)`.
- You should provide a name or caption for each value in the final output considering the question and the input context."
- Dont create any sample data in order to answer to the user question.
- You should print the final data structure.
- You should save the final data structure at the specified path with a proper filename.
- You should output the final data structure as a final output.
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
    
    data: str = Field(
        ...,
        description="The final data structure as a final output.",
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
        # print(code)
        # output = str(self.python_tool.run(code))
        
        # if output == "":
        #     return "Your code is executed successfully"
        # else:
        #     return output
        try:
            result = self.python_tool.run(code)
        except Exception as e:
            return f"Failed to execute. Error: {repr(e)}"
        return result
        
python_repl = PythonREPL() 
      

def get_data_preparation_tools(llm: ChatOpenAI, log_path):
    """
   
    Args:
        question (str): The question.
        context list(str)
    Returns:
        dataframe: the dataframe that is needed for a plot genration task.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{question}"),
            ("user", "{context}"),
            
        ]
    )
    
    extractor = create_structured_output_runnable(ExecuteCode, llm, prompt)


    def data_preparation(
        question: str,
        context: Union[str, List[str],dict] = None,
        config: Optional[RunnableConfig] = None,
    ):
       
        print("context-first:", context,type(context))
        context_str= str(context).strip()
        # context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
        #     context= context_str.strip()
        # )
        # if 'data' in context:
        #     context=context['data']
        context_str += f"Save the generated data to the following directory: {log_path} and output the final data structure in data filed"
        chain_input = {"question": question,"context":context_str}
        # chain_input["context"] = [SystemMessage(content=context)]
                       
        code_model = extractor.invoke(chain_input, config)

        if code_model.code=='':
            return code_model.reasoning 
        codeExecution_result = python_repl.run(code_model.code)
        if "Error" in codeExecution_result:
            _error_handiling_prompt=f"Something went wrong on executing Code: `{code_model.code}`. This is the error I got: `{codeExecution_result}`. \\ Can you fixed the problem and write the fixed python code?"
            chain_input["info"] =[HumanMessage(content= _error_handiling_prompt)]
            code_model = extractor.invoke(chain_input)
            try:
                return code_model.data
            except Exception as e:
                return repr(e)
        else:
            # extract data from the code
           
            return code_model.data


    return StructuredTool.from_function(
        name = "data_preparation",
        func = data_preparation,
        description=_DESCRIPTION,
    )


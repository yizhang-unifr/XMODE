from typing import List, Optional
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
import sqlite3
import re
import json
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union,List

from src.utils import _get_db_schema


_meta_data="""
The column 'title' in the table contains the title of the artwork. Type: TEXT.
The column 'inception' in the table contains the date when the artwork was created. Type: DATETIME.
The column 'movement' in the table contains the art movement that the artwork belongs to. Type: TEXT.
The column 'genre' in the table contains the genre of the artwork. Type: TEXT.
The column 'image_url' in the table contains the URL of the image of the artwork. Type: TEXT.
The column 'img_path' in the table contains the path to the image of the artwork in the local system. Type: TEXT.
"""

_DESCRIPTION = (
    "text2SQL(problem: str, context: Optional[Union[str,list[str]]])-->str\n"
    "The input for this tools should be `problem` as a textual question\n"
    # Context specific rules below
    " - You can optionally provide a list of strings as `context` to help the agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    "In the 'context' you could add any other information that you think is required to generate te SQL code. It can be the information from previous taks.\n" 
    "This tools is able to translate the question to the SQL code considering the database information.\n"
    "The SQL code can be executed using sqlite3 library.\n"
    "Use the output of running generated SQL code to answer the question.\n"
    
)


_SYSTEM_PROMPT = """  
You are a database expert. Generate a SQL query given the following user question, database information and other context that you receive.
You should analyse the question, context and the database schema and come with the executabel sqlite3 query. 
Provide all the required information in the SQL code to answer the original user question that may required in other tasks utilizing the relevant database schema.
Ensure you include all necessary information, including columns used for filtering, especially when the task involves plotting or data exploration.
This must be taken into account when performing any time-based data queries or analyses.
if the question asks for information that is not found in the database schema, you must retrieve the `ima_path` for image analysis task.
Translate a text question into a SQL query that can be executed on the SQLite database.
List of Businnes Roles to take into account during the translation task:
1- To calculate century from inception field use : (CAST(strftime('%Y', inception) AS INTEGER) - 1) / 100 + 1
....
"""
#If you want to consider "now" or "current_time", then replace them with strftime('2105-12-31 23:59:59').
_ADDITIONAL_CONTEXT_PROMPT = """
"""


class ExecuteCode(BaseModel):

    reasoning: str = Field(
        ...,
        description="The reasoning behind the SQL expression, including how context is included, if applicable.",
    )

    SQL: str = Field(
        ...,
        description="The SQL Code that can be runnable on the corresponding database ",
    )
    

# def _clean_query(q):
#     res=q.replace('\n'," ").replace('%y','%Y').replace('current_time',"strftime('2105-12-31 23:59:59')").replace('now',"strftime('2105-12-31 23:59:59')")
#     return res


# def _parse_input(input_text: str) -> dict:
#     """
#     Args:
#         input_text (str): The text containing input in dict format.

#     Returns:
#         dict: The parsed sub-questions as a structured dictionary.
#     """
#     try:
#         # Remove any potential non-JSON text and extract the JSON part
#         json_match = re.search(r'\{.*\}', input_text, re.DOTALL)
#         if json_match:
#             input_text_json = json_match.group(0)
#             input_text = json.loads(input_text_json)
#         else:
#             raise ValueError("No JSON object found in the text")
#         return input_text
#     except (json.JSONDecodeError, ValueError) as e:
#         print(f"Error parsing sub-questions: {e}")
#         return {
#             "tables":[],
#             "columns":[]
#         }
    
def _execute_sql_query(query: str, db_path: str, as_dict=True) -> Dict[str, Any]:
    try:
        if as_dict:
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            # print("SQL:",change_current_time(query))
            cur.execute(query)
            results = [dict(row) for row in cur.fetchall()]
            # print("results of SQL",results)
        else:
            engine = create_engine(f'sqlite:///{db_path}')
            database = SQLDatabase(engine, sample_rows_in_table_info=0)
            results = database.run(query)
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}



def get_text2SQL_tools(llm: ChatOpenAI, db_path:str):
    """
    Provide the SQL code from a given question.

    Args:
        raw_question (str): The raw user question.
        schema (str): The database information such as the Tables and Columns.

    Returns:
        results (SQL QUERY str)
    """

    _db_schema = _get_db_schema(db_path)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            ("user", f"{_db_schema}/n{_meta_data}"),
            MessagesPlaceholder(variable_name="info", optional=True),
        ]
    )
    # extractor = create_structured_output_runnable(ExecuteCode, llm, prompt)
    extractor= prompt | llm.with_structured_output(ExecuteCode)
    
    def text2SQL(
        problem: str,
        context: Optional[Union[str,List[str]]] = None,
    ):
        #tables_columns=_parse_input(tables_columns)
        chain_input = {"problem": problem}
        if context:
            if isinstance(context,list):
                context_str = "\n".join(context)
            else:
                context_str = context
            chain_input["info"] = [HumanMessage(content=context_str)]
        code_model = extractor.invoke(chain_input)
        try:
            return _execute_sql_query(code_model.SQL, db_path)
        except Exception as e:
            # self_debugging 
            err = repr(e)
            _error_handiling_prompt=f"Something went wrong on executing SQL: `{code_model.SQL}`. This is the error I got: `{err}`. \\ Can you fixed the problem and write the fixed SQL code?"
            chain_input["info"] =[HumanMessage(content= [context_str, _error_handiling_prompt])]
            code_model = extractor.invoke(chain_input)
            try:
                return _execute_sql_query(code_model.SQL, db_path)
            except Exception as e:
                return repr(e)

    return StructuredTool.from_function(
        name="text2SQL",
        func=text2SQL,
        description=_DESCRIPTION,
    )



'''
 chat_thread = []
        i = 0
        while True:
            try:
                func, dtype, func_str = self.get_func(explanation, ds.data_frame[column][:10],
                                                      chat_thread=chat_thread, column=column, new_column=new_name)
                df = ds.data_frame.copy()
                df[new_name] = df[column].apply(func).astype(dtype)
                return df, func_str
            except Exception as e:
                if i >= 3:
                    raise ExecutionError(description="Python tool failed. Use another tool!")
                chat_thread = self.handle_errors(chat_thread=chat_thread, error=e, request=explanation)
                i += 1
'''